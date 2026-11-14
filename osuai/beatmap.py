"""Beatmap parsing, timing/SV resolution, L/P/B/C slider geometry, and mods."""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Literal

PLAYFIELD_W = 512.0
PLAYFIELD_H = 384.0

KEY_M1 = 1
KEY_M2 = 2
KEY_K1 = 4
KEY_K2 = 8
KEY_SMOKE = 16
KEY_HIT_MASK = KEY_M1 | KEY_M2 | KEY_K1 | KEY_K2

MOD_BITS: dict[str, int] = {
    "NM": 0,
    "NF": 1,
    "EZ": 2,
    "TD": 4,
    "HD": 8,
    "HR": 16,
    "SD": 32,
    "DT": 64,
    "RX": 128,
    "HT": 256,
    "NC": 512 | 64,  # NC sets DT bit in .osr
    "FL": 1024,
    "AT": 2048,
    "SO": 4096,
    "AP": 8192,
    "PF": 16384 | 32,  # PF sets SD bit in .osr
    "SCOREV2": 536870912,
}

SUPPORTED_MODS = {"NM", "NF", "EZ", "HD", "HR", "SD", "DT", "HT", "NC", "FL", "PF"}


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        raise IndexError(idx)

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def dist(self, other: Vec2) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def lerp(self, other: Vec2, t: float) -> Vec2:
        return Vec2(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)

    def clamp(self, w: float = PLAYFIELD_W, h: float = PLAYFIELD_H) -> Vec2:
        return Vec2(min(max(self.x, 0.0), w), min(max(self.y, 0.0), h))


@dataclass(slots=True)
class TimingPoint:
    time: float
    beat_length: float
    uninherited: bool
    sv: float
    meter: int = 4


@dataclass(slots=True)
class Frame:
    time: float
    x: float
    y: float
    keys: int = 0

    @property
    def pos(self) -> Vec2:
        return Vec2(self.x, self.y)


@dataclass(slots=True)
class HitObject:
    kind: Literal["circle", "slider", "spinner"]
    time: float
    pos: Vec2
    radius: float
    end_time: float
    end_pos: Vec2
    # slider fields
    path: list[Vec2] = field(default_factory=list)
    cum_lengths: list[float] = field(default_factory=list)
    path_length: float = 0.0
    slides: int = 1
    span_duration: float = 0.0
    tick_times: list[float] = field(default_factory=list)
    tick_positions: list[Vec2] = field(default_factory=list)
    repeat_times: list[float] = field(default_factory=list)
    repeat_positions: list[Vec2] = field(default_factory=list)
    # spinner fields
    duration: float = 0.0
    # stacking metadata
    stack_count: int = 0

    def slider_pos_at_distance(self, d: float) -> Vec2:
        """Position on single forward span at arc-length distance d in [0, path_length]."""
        if not self.path:
            return self.pos
        if len(self.path) == 1 or self.path_length <= 0.0 or d <= 0.0:
            return self.path[0]
        if d >= self.path_length:
            return self.path[-1]
        idx = bisect.bisect_right(self.cum_lengths, d) - 1
        idx = max(0, min(idx, len(self.path) - 2))
        l0 = self.cum_lengths[idx]
        l1 = self.cum_lengths[idx + 1]
        seg = l1 - l0
        t = (d - l0) / seg if seg > 1e-9 else 0.0
        return self.path[idx].lerp(self.path[idx + 1], t)

    def slider_pos_at_time(self, t: float) -> Vec2:
        """Position on slider across repeats at map time t."""
        if self.kind != "slider" or not self.path or self.span_duration <= 0.0:
            return self.pos
        if t <= self.time:
            return self.path[0]
        if t >= self.end_time:
            return self.end_pos
        elapsed = t - self.time
        span = int(elapsed // self.span_duration)
        if span >= self.slides:
            return self.end_pos
        local_frac = (elapsed - span * self.span_duration) / self.span_duration
        local_frac = min(max(local_frac, 0.0), 1.0)
        dist = local_frac * self.path_length if span % 2 == 0 else (1.0 - local_frac) * self.path_length
        return self.slider_pos_at_distance(dist)


@dataclass(slots=True)
class Difficulty:
    hp: float = 5.0
    cs: float = 5.0
    od: float = 5.0
    ar: float = 5.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0
    mods: tuple[str, ...] = ("NM",)
    mods_bitmask: int = 0

    @property
    def circle_radius(self) -> float:
        return 54.4 - 4.48 * self.cs

    @property
    def follow_radius(self) -> float:
        return 2.4 * self.circle_radius

    @property
    def w300(self) -> float:
        return 80.0 - 6.0 * self.od

    @property
    def w100(self) -> float:
        return 140.0 - 8.0 * self.od

    @property
    def w50(self) -> float:
        return 200.0 - 10.0 * self.od

    @property
    def preempt(self) -> float:
        return 1800.0 - 120.0 * self.ar if self.ar <= 5.0 else 1200.0 - 150.0 * (self.ar - 5.0)

    @property
    def clock_rate(self) -> float:
        if "DT" in self.mods or "NC" in self.mods:
            return 1.5
        if "HT" in self.mods:
            return 0.75
        return 1.0


@dataclass(slots=True)
class Beatmap:
    md5: str
    title: str
    artist: str
    creator: str
    version: str
    beatmap_id: int
    mode: int
    stack_leniency: float
    base_difficulty: Difficulty
    difficulty: Difficulty
    timing_points: list[TimingPoint]
    objects: list[HitObject]
    path: Path | None = None
    raw_bytes: bytes = b""
    warnings: list[str] = field(default_factory=list)

    @property
    def max_combo(self) -> int:
        total = 0
        for obj in self.objects:
            if obj.kind == "slider":
                total += 2 + len(obj.tick_times) + len(obj.repeat_times)
            else:
                total += 1
        return total

    def bpm_range(self) -> tuple[float, float]:
        bpms = [60000.0 / tp.beat_length for tp in self.timing_points if tp.uninherited and tp.beat_length > 0]
        if not bpms:
            return (120.0, 120.0)
        scale = self.difficulty.clock_rate
        return (min(bpms) * scale, max(bpms) * scale)

    def timing_at(self, t: float) -> tuple[float, float]:
        """Return (beat_length, sv) active at map time t."""
        return resolve_timing_at(self.timing_points, t)

    def with_mods(self, mods: Iterable[str] | str | int, hr_flip_y: bool = False) -> Beatmap:
        mod_tuple, bitmask = parse_mods(mods)
        new_diff = apply_mods_to_difficulty(self.base_difficulty, mod_tuple, bitmask)
        r = new_diff.circle_radius
        flip = hr_flip_y and ("HR" in mod_tuple)
        new_objs: list[HitObject] = []
        for obj in self.objects:
            if not flip:
                new_objs.append(replace(obj, radius=r))
            else:
                new_objs.append(
                    replace(
                        obj,
                        pos=Vec2(obj.pos.x, PLAYFIELD_H - obj.pos.y),
                        end_pos=Vec2(obj.end_pos.x, PLAYFIELD_H - obj.end_pos.y),
                        radius=r,
                        path=[Vec2(p.x, PLAYFIELD_H - p.y) for p in obj.path],
                        tick_positions=[Vec2(p.x, PLAYFIELD_H - p.y) for p in obj.tick_positions],
                        repeat_positions=[Vec2(p.x, PLAYFIELD_H - p.y) for p in obj.repeat_positions],
                    )
                )
        return replace(self, difficulty=new_diff, objects=new_objs)


def parse_mods(mods: Iterable[str] | str | int | None) -> tuple[tuple[str, ...], int]:
    """Parse mod names, comma-separated string, or bitmask into (canonical_tuple, bitmask)."""
    if mods is None:
        return (("NM",), 0)
    if isinstance(mods, int):
        if mods == 0:
            return (("NM",), 0)
        names: list[str] = []
        if mods & 512:
            names.append("NC")
        elif mods & 64:
            names.append("DT")
        if mods & 256:
            names.append("HT")
        if mods & 16:
            names.append("HR")
        if mods & 2:
            names.append("EZ")
        if mods & 8:
            names.append("HD")
        if mods & 1024:
            names.append("FL")
        if mods & 1:
            names.append("NF")
        if mods & 16384:
            names.append("PF")
        elif mods & 32:
            names.append("SD")
        clean_mask = sum(MOD_BITS[m] for m in names)
        return (tuple(names) if names else ("NM",), clean_mask)

    if isinstance(mods, str):
        raw_items = [m.strip().upper() for m in mods.replace("+", ",").split(",") if m.strip()]
    else:
        raw_items = [str(m).strip().upper() for m in mods if str(m).strip()]

    if not raw_items or raw_items == ["NM"]:
        return (("NM",), 0)

    clean: list[str] = []
    bitmask = 0
    for item in raw_items:
        if item == "NM":
            continue
        if item not in SUPPORTED_MODS:
            raise ValueError(f"Unsupported mod: {item}")
        if item not in clean:
            clean.append(item)
            bitmask |= MOD_BITS[item]

    if "HR" in clean and "EZ" in clean:
        raise ValueError("Incompatible mods: HR and EZ")
    if ("DT" in clean or "NC" in clean) and "HT" in clean:
        raise ValueError("Incompatible mods: DT/NC and HT")
    if "NF" in clean and ("SD" in clean or "PF" in clean):
        raise ValueError("Incompatible mods: NF and SD/PF")

    return (tuple(clean) if clean else ("NM",), bitmask)


def apply_mods_to_difficulty(base: Difficulty, mods: tuple[str, ...], bitmask: int) -> Difficulty:
    hp, cs, od, ar = base.hp, base.cs, base.od, base.ar
    if "HR" in mods:
        cs = min(cs * 1.4, 10.0)
        ar = min(ar * 1.4, 10.0)
        od = min(od * 1.4, 10.0)
        hp = min(hp * 1.4, 10.0)
    elif "EZ" in mods:
        cs *= 0.5
        ar *= 0.5
        od *= 0.5
        hp *= 0.5
    return Difficulty(
        hp=hp,
        cs=cs,
        od=od,
        ar=ar,
        slider_multiplier=base.slider_multiplier,
        slider_tick_rate=base.slider_tick_rate,
        mods=mods,
        mods_bitmask=bitmask,
    )


def resolve_timing_at(timing_points: list[TimingPoint], t: float) -> tuple[float, float]:
    """Resolve effective (beat_length, sv) at time t (red uninherited line resets sv to 1.0)."""
    if not timing_points:
        return (500.0, 1.0)
    beat_length: float | None = None
    sv = 1.0
    for tp in timing_points:
        if tp.time <= t + 1e-6:
            if tp.uninherited:
                beat_length = tp.beat_length
                sv = 1.0
            else:
                sv = tp.sv
        else:
            break
    if beat_length is None:
        for tp in timing_points:
            if tp.uninherited:
                beat_length = tp.beat_length
                break
        if beat_length is None:
            beat_length = 500.0
    return (max(beat_length, 1e-3), max(sv, 1e-3))


def _sample_linear(pts: list[Vec2]) -> list[Vec2]:
    return list(pts) if len(pts) >= 2 else (pts * 2 if pts else [Vec2(0.0, 0.0)])


def _de_casteljau(pts: list[Vec2], t: float) -> Vec2:
    work = list(pts)
    n = len(work)
    for r in range(1, n):
        for i in range(n - r):
            work[i] = work[i].lerp(work[i + 1], t)
    return work[0]


def _sample_bezier_segment(pts: list[Vec2], samples_per_seg: int = 64) -> list[Vec2]:
    if len(pts) <= 1:
        return list(pts)
    if len(pts) == 2:
        return [pts[0], pts[1]]
    n = max(samples_per_seg, int(sum(pts[i].dist(pts[i + 1]) for i in range(len(pts) - 1)) / 2.0))
    n = min(max(n, 16), 256)
    return [_de_casteljau(pts, i / n) for i in range(n + 1)]


def _sample_bezier(pts: list[Vec2]) -> list[Vec2]:
    if len(pts) <= 2:
        return _sample_linear(pts)
    segments: list[list[Vec2]] = []
    cur: list[Vec2] = [pts[0]]
    for p in pts[1:]:
        if p.dist(cur[-1]) < 1e-4:
            if len(cur) >= 2:
                segments.append(cur)
            cur = [p]
        else:
            cur.append(p)
    if len(cur) >= 2:
        segments.append(cur)
    if not segments:
        return _sample_linear(pts)

    out: list[Vec2] = []
    for seg in segments:
        sampled = _sample_bezier_segment(seg)
        if out:
            out.extend(sampled[1:])
        else:
            out.extend(sampled)
    return out


def _sample_perfect_circle(pts: list[Vec2]) -> list[Vec2]:
    if len(pts) != 3:
        return _sample_bezier(pts)
    p0, p1, p2 = pts
    d = 2.0 * (p0.x * (p1.y - p2.y) + p1.x * (p2.y - p0.y) + p2.x * (p0.y - p1.y))
    if abs(d) < 1e-5:
        return _sample_linear(pts)

    p0_sq = p0.x * p0.x + p0.y * p0.y
    p1_sq = p1.x * p1.x + p1.y * p1.y
    p2_sq = p2.x * p2.x + p2.y * p2.y

    cx = (p0_sq * (p1.y - p2.y) + p1_sq * (p2.y - p0.y) + p2_sq * (p0.y - p1.y)) / d
    cy = (p0_sq * (p2.x - p1.x) + p1_sq * (p0.x - p2.x) + p2_sq * (p1.x - p0.x)) / d
    center = Vec2(cx, cy)
    r = center.dist(p0)
    if r > 1e5 or r < 1e-4:
        return _sample_linear(pts)

    a0 = math.atan2(p0.y - cy, p0.x - cx)
    a1 = math.atan2(p1.y - cy, p1.x - cx)
    a2 = math.atan2(p2.y - cy, p2.x - cx)

    def ccw_delta(start: float, end: float) -> float:
        return (end - start) % (2.0 * math.pi)

    d01 = ccw_delta(a0, a1)
    d02 = ccw_delta(a0, a2)
    sweep = d02 if d01 <= d02 else d02 - 2.0 * math.pi
    arc_len = abs(sweep) * r
    steps = min(max(int(arc_len / 2.0), 24), 512)
    return [
        Vec2(cx + r * math.cos(a0 + sweep * (i / steps)), cy + r * math.sin(a0 + sweep * (i / steps)))
        for i in range(steps + 1)
    ]


def _catmull_point(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: float) -> Vec2:
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2.0 * p1.x)
        + (-p0.x + p2.x) * t
        + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
        + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3
    )
    y = 0.5 * (
        (2.0 * p1.y)
        + (-p0.y + p2.y) * t
        + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
        + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3
    )
    return Vec2(x, y)


def _sample_catmull(pts: list[Vec2], steps_per_seg: int = 32) -> list[Vec2]:
    if len(pts) <= 2:
        return _sample_linear(pts)
    out: list[Vec2] = []
    n = len(pts)
    for i in range(n - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < n else p2
        start_step = 1 if out else 0
        for s in range(start_step, steps_per_seg + 1):
            out.append(_catmull_point(p0, p1, p2, p3, s / steps_per_seg))
    return out


def build_slider_path(start: Vec2, curve_str: str, target_length: float) -> tuple[list[Vec2], list[float], float]:
    """Parse slider curve string and normalize polyline to target_length."""
    parts = curve_str.split("|")
    curve_type = parts[0].strip().upper() if parts else "L"
    ctrl: list[Vec2] = [start]
    for token in parts[1:]:
        if ":" not in token:
            continue
        xs, ys = token.split(":", 1)
        pt = Vec2(float(xs), float(ys))
        if len(ctrl) == 1 and pt.dist(start) < 1e-3:
            continue
        ctrl.append(pt)

    if target_length <= 0.0 or len(ctrl) < 2:
        return ([start], [0.0], 0.0)

    if curve_type == "L":
        raw = _sample_linear(ctrl)
    elif curve_type == "P":
        raw = _sample_perfect_circle(ctrl)
    elif curve_type == "C":
        raw = _sample_catmull(ctrl)
    else:
        raw = _sample_bezier(ctrl)

    cleaned: list[Vec2] = [raw[0]]
    cum: list[float] = [0.0]
    for p in raw[1:]:
        d = p.dist(cleaned[-1])
        if d > 1e-6:
            cleaned.append(p)
            cum.append(cum[-1] + d)

    if len(cleaned) == 1 or cum[-1] <= 1e-6:
        return ([start], [0.0], 0.0)

    total_raw = cum[-1]
    if total_raw >= target_length:
        trimmed: list[Vec2] = [cleaned[0]]
        trimmed_cum: list[float] = [0.0]
        for i in range(1, len(cleaned)):
            if cum[i] < target_length:
                trimmed.append(cleaned[i])
                trimmed_cum.append(cum[i])
            else:
                prev_l = cum[i - 1]
                seg_l = cum[i] - prev_l
                frac = (target_length - prev_l) / seg_l if seg_l > 1e-9 else 0.0
                end_pt = cleaned[i - 1].lerp(cleaned[i], frac)
                trimmed.append(end_pt)
                trimmed_cum.append(target_length)
                break
        return (trimmed, trimmed_cum, target_length)

    scale = target_length / total_raw
    scaled_cum = [c * scale for c in cum]
    scaled_cum[-1] = target_length
    return (cleaned, scaled_cum, target_length)


def apply_stacking(objects: list[HitObject], ar_preempt: float, stack_leniency: float) -> None:
    """Apply osu!stable's reverse-pass stacking offset (-3, -3 osu!px per stack)."""
    if not objects or stack_leniency <= 0.0:
        return
    stack_threshold = ar_preempt * stack_leniency
    stack_dist = 3.0
    for o in objects:
        o.stack_count = 0

    n = len(objects)
    for i in range(n - 1, 0, -1):
        obj_i = objects[i]
        if obj_i.kind == "spinner" or obj_i.stack_count != 0:
            continue
        for j in range(i - 1, -1, -1):
            obj_j = objects[j]
            if obj_j.kind == "spinner":
                continue
            if obj_i.time - obj_j.end_time > stack_threshold:
                break
            if obj_j.kind == "slider" and obj_j.end_pos.dist(obj_i.pos) < stack_dist:
                offset = obj_i.stack_count - obj_j.stack_count + 1
                for k in range(j + 1, i + 1):
                    if obj_j.end_pos.dist(objects[k].pos) < stack_dist:
                        objects[k].stack_count -= offset
                break
            if obj_j.pos.dist(obj_i.pos) < stack_dist:
                obj_j.stack_count = obj_i.stack_count + 1
                obj_i = obj_j

    for obj in objects:
        if obj.stack_count != 0:
            delta = Vec2(-3.0 * obj.stack_count, -3.0 * obj.stack_count)
            obj.pos = obj.pos + delta
            obj.end_pos = obj.end_pos + delta
            if obj.kind == "slider":
                obj.path = [p + delta for p in obj.path]
                obj.tick_positions = [p + delta for p in obj.tick_positions]
                obj.repeat_positions = [p + delta for p in obj.repeat_positions]


def parse_osu(source: str | Path | bytes, mods: Iterable[str] | str | int | None = None, enable_stacking: bool = False) -> Beatmap:
    """Parse a .osu beatmap from file path, raw bytes, or text string."""
    path: Path | None = None
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).exists()):
        path = Path(source)
        raw_bytes = path.read_bytes()
    elif isinstance(source, bytes):
        raw_bytes = source
    elif isinstance(source, str):
        raw_bytes = source.encode("utf-8")
    else:
        raise TypeError(f"Unsupported beatmap source: {type(source)}")

    md5 = hashlib.md5(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    first_line = text.splitlines()[0].strip() if text else ""
    format_version = 14
    if "osu file format v" in first_line:
        try:
            format_version = int(first_line.split("osu file format v", 1)[1].strip())
        except ValueError:
            format_version = 14

    general: dict[str, str] = {}
    metadata: dict[str, str] = {}
    diff_raw: dict[str, float] = {}
    timing_lines: list[str] = []
    object_lines: list[str] = []
    warnings: list[str] = []

    section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue

        if section in ("General", "Metadata", "Difficulty"):
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if section == "General":
                general[k] = v
            elif section == "Metadata":
                metadata[k] = v
            elif section == "Difficulty":
                try:
                    diff_raw[k] = float(v)
                except ValueError:
                    pass
        elif section == "TimingPoints":
            timing_lines.append(line)
        elif section == "HitObjects":
            object_lines.append(line)

    mode = int(general.get("Mode", "0"))
    if mode != 0:
        raise ValueError(f"Unsupported game mode: Mode={mode} (only Mode=0 osu!standard is supported)")

    hp = diff_raw.get("HPDrainRate", 5.0)
    cs = diff_raw.get("CircleSize", 5.0)
    od = diff_raw.get("OverallDifficulty", 5.0)
    ar = diff_raw.get("ApproachRate", od)
    slider_mult = max(diff_raw.get("SliderMultiplier", 1.4), 1e-3)
    slider_tick_rate = max(diff_raw.get("SliderTickRate", 1.0), 1e-3)
    stack_leniency = float(general.get("StackLeniency", "0.7"))

    base_diff = Difficulty(
        hp=hp,
        cs=cs,
        od=od,
        ar=ar,
        slider_multiplier=slider_mult,
        slider_tick_rate=slider_tick_rate,
    )
    mod_tuple, mod_bitmask = parse_mods(mods)
    active_diff = apply_mods_to_difficulty(base_diff, mod_tuple, mod_bitmask)

    indexed_tps: list[tuple[float, int, TimingPoint]] = []
    for idx, tline in enumerate(timing_lines):
        fields = [f.strip() for f in tline.split(",")]
        if len(fields) < 2:
            continue
        t_val = float(fields[0])
        bl_val = float(fields[1])
        meter = int(float(fields[2])) if len(fields) >= 3 and fields[2] else 4
        if len(fields) >= 7 and fields[6] != "":
            uninherited = int(float(fields[6])) != 0
        else:
            uninherited = bl_val > 0.0

        if uninherited:
            beat_len = bl_val if bl_val > 0.0 else 500.0
            sv = 1.0
        else:
            beat_len = bl_val
            sv = min(max(-100.0 / bl_val, 0.1), 10.0) if bl_val < 0.0 else 1.0

        tp = TimingPoint(time=t_val, beat_length=beat_len, uninherited=uninherited, sv=sv, meter=meter)
        indexed_tps.append((t_val, idx, tp))

    if not indexed_tps:
        raise ValueError("Beatmap has no timing points")

    # At identical timestamps, process uninherited (red) point before inherited (green) point
    indexed_tps.sort(key=lambda item: (item[0], 0 if item[2].uninherited else 1, item[1]))
    timing_points = [item[2] for item in indexed_tps]
    first_uninherited_time = next((tp.time for tp in timing_points if tp.uninherited), None)
    if first_uninherited_time is None:
        warnings.append("No uninherited timing point found; defaulting to 500 ms/beat")

    radius = active_diff.circle_radius
    objects: list[HitObject] = []
    for oline in object_lines:
        parts = [p.strip() for p in oline.split(",")]
        if len(parts) < 4:
            continue
        x = float(parts[0])
        y = float(parts[1])
        obj_time = float(parts[2])
        obj_type = int(float(parts[3]))
        pos = Vec2(x, y)

        if first_uninherited_time is not None and obj_time < first_uninherited_time:
            warnings.append(f"Object at {obj_time}ms precedes first uninherited timing point ({first_uninherited_time}ms)")

        if obj_type & 128:
            raise ValueError("Mania hold note (type bit 128) encountered in standard beatmap")

        if obj_type & 1:
            objects.append(
                HitObject(
                    kind="circle",
                    time=obj_time,
                    pos=pos,
                    radius=radius,
                    end_time=obj_time,
                    end_pos=pos,
                )
            )
        elif obj_type & 2:
            if len(parts) < 8:
                continue
            curve_str = parts[5]
            slides = max(int(float(parts[6])), 1)
            raw_length = max(float(parts[7]), 0.0)
            beat_len, sv = resolve_timing_at(timing_points, obj_time)
            path_pts, cum_lens, length = build_slider_path(pos, curve_str, raw_length)

            denom = slider_mult * 100.0 * sv
            span_duration = (length / denom) * beat_len if (length > 0.0 and denom > 0.0) else 0.0
            end_time = obj_time + span_duration * slides
            end_pos = path_pts[-1] if (slides % 2 == 1) else path_pts[0]

            tick_dist = (
                (slider_mult * 100.0) / slider_tick_rate
                if format_version < 8
                else denom / slider_tick_rate
            )
            min_dist_from_end = 10.0 * denom / beat_len
            tick_times: list[float] = []
            tick_positions: list[Vec2] = []
            repeat_times: list[float] = []
            repeat_positions: list[Vec2] = []

            if length > 0.0 and span_duration > 0.0:
                base_tick_dists: list[float] = []
                if tick_dist > 1e-3:
                    k = 1
                    while k * tick_dist <= length - min_dist_from_end:
                        base_tick_dists.append(k * tick_dist)
                        k += 1

                for s in range(slides):
                    span_start = obj_time + s * span_duration
                    forward = (s % 2 == 0)
                    span_dists = base_tick_dists if forward else [length - d for d in reversed(base_tick_dists)]
                    for d_along_span in span_dists:
                        t_tick = span_start + (d_along_span / length) * span_duration
                        p_dist = d_along_span if forward else (length - d_along_span)
                        p_tick = _interp_polyline(path_pts, cum_lens, length, p_dist)
                        tick_times.append(t_tick)
                        tick_positions.append(p_tick)

                    if s < slides - 1:
                        r_time = span_start + span_duration
                        r_pos = path_pts[-1] if forward else path_pts[0]
                        repeat_times.append(r_time)
                        repeat_positions.append(r_pos)

            objects.append(
                HitObject(
                    kind="slider",
                    time=obj_time,
                    pos=pos,
                    radius=radius,
                    end_time=end_time,
                    end_pos=end_pos,
                    path=path_pts,
                    cum_lengths=cum_lens,
                    path_length=length,
                    slides=slides,
                    span_duration=span_duration,
                    tick_times=tick_times,
                    tick_positions=tick_positions,
                    repeat_times=repeat_times,
                    repeat_positions=repeat_positions,
                    duration=end_time - obj_time,
                )
            )
        elif obj_type & 8:
            end_t = float(parts[5]) if len(parts) >= 6 else obj_time
            end_t = max(end_t, obj_time)
            center = Vec2(256.0, 192.0)
            objects.append(
                HitObject(
                    kind="spinner",
                    time=obj_time,
                    pos=center,
                    radius=radius,
                    end_time=end_t,
                    end_pos=center,
                    duration=end_t - obj_time,
                )
            )

    objects.sort(key=lambda o: o.time)
    if enable_stacking:
        apply_stacking(objects, active_diff.preempt, stack_leniency)

    return Beatmap(
        md5=md5,
        title=metadata.get("Title", "Unknown"),
        artist=metadata.get("Artist", "Unknown"),
        creator=metadata.get("Creator", "Unknown"),
        version=metadata.get("Version", "Normal"),
        beatmap_id=int(metadata.get("BeatmapID", "0") or 0),
        mode=mode,
        stack_leniency=stack_leniency,
        base_difficulty=base_diff,
        difficulty=active_diff,
        timing_points=timing_points,
        objects=objects,
        path=path,
        raw_bytes=raw_bytes,
        warnings=warnings,
    )


def _interp_polyline(path: list[Vec2], cum_lengths: list[float], total_len: float, d: float) -> Vec2:
    if not path:
        return Vec2(0.0, 0.0)
    if len(path) == 1 or total_len <= 0.0 or d <= 0.0:
        return path[0]
    if d >= total_len:
        return path[-1]
    idx = bisect.bisect_right(cum_lengths, d) - 1
    idx = max(0, min(idx, len(path) - 2))
    l0 = cum_lengths[idx]
    l1 = cum_lengths[idx + 1]
    seg = l1 - l0
    t = (d - l0) / seg if seg > 1e-9 else 0.0
    return path[idx].lerp(path[idx + 1], t)
