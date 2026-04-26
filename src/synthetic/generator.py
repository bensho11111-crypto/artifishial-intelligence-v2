"""
src/synthetic/generator.py

Generates synthetic Tick streams from a FloorModel.

Produces the same data types as real hardware so the rest of the
pipeline (fusion, world_state, server) doesn't know the difference.
"""
from __future__ import annotations
import math
import random
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from ticks.models import Tick, SonarTick, GpsTick
from synthetic.floor import FloorModel


# ── Simulation parameters ─────────────────────────────────────────────────────

START_LAT     = 33.9003
START_LON     = -117.5012
START_HEADING = 52.0
SPEED_KTS     = 3.5
WATER_TEMP_C  = 18.3
GPS_HZ        = 1
SONAR_HZ      = 5
ECHO_SIZE     = 512
MAX_RANGE_M   = 60.0
# Beam half-angles differ by frequency: lower freq → wider beam for same aperture.
# 83 kHz: ~12° half-angle;  200 kHz: ~6° half-angle (typical Lowrance/Garmin specs).
_TAN_HALF_LF  = math.tan(math.radians(12.0))
_TAN_HALF_HF  = math.tan(math.radians(6.0))

# Two-way acoustic attenuation coefficients (Np/m).
# Higher frequency → faster amplitude decay with depth.
#   83 kHz  ≈ 0.026 dB/m  → 0.003 Np/m
#  200 kHz  ≈ 0.070 dB/m  → 0.008 Np/m
ALPHA_LF = 0.003   # 83 kHz
ALPHA_HF = 0.008   # 200 kHz


# ── Fish schools ──────────────────────────────────────────────────────────────

@dataclass
class FishSchool:
    east_m:   float
    north_m:  float
    depth_m:  float
    radius_m: float
    density:  float
    species:  str
    # Sinusoidal movement
    amp_e:  float = 0.0
    amp_n:  float = 0.0
    freq:   float = 0.0
    phase:  float = 0.0
    # Acoustic properties (per-species, set by _make_schools)
    ts_lf:  float = 0.70   # 83 kHz target strength scalar
    ts_hf:  float = 0.70   # 200 kHz target strength scalar
    n_fish: int   = 15     # individual fish count for arch simulation
    shape:  str   = "sphere"  # "sphere" (pelagic) or "disk" (demersal)

    def at(self, t: float) -> "FishSchool":
        from dataclasses import replace
        return replace(
            self,
            east_m  = self.east_m  + self.amp_e * math.sin(self.freq * t + self.phase),
            north_m = self.north_m + self.amp_n * math.cos(self.freq * t + self.phase + 0.5),
        )


# Per-species acoustic signatures.
# The LF/HF ratio is the primary ML discriminant:
#   physoclistous fish (bass, trout) have a resonant swimbladder that back-scatters
#   strongly at low frequency → high LF/HF ratio.
#   physostomous / small-swimbladder fish (carp, bream) scatter more uniformly
#   across frequencies → low LF/HF ratio.
SPECIES_ACOUSTICS = {
    "bass":  {"ts_lf": 0.85, "ts_hf": 0.55, "n_fish": 25, "shape": "sphere"},
    "trout": {"ts_lf": 0.72, "ts_hf": 0.62, "n_fish": 20, "shape": "sphere"},
    "carp":  {"ts_lf": 0.33, "ts_hf": 0.58, "n_fish": 15, "shape": "disk"},
    "bream": {"ts_lf": 0.45, "ts_hf": 0.70, "n_fish": 30, "shape": "disk"},
}

SPECIES_NAMES = {
    "bass":  "largemouth bass",
    "trout": "rainbow trout",
    "carp":  "common carp",
    "bream": "bluegill bream",
}


def _enu(lat, lon):
    R = 6_371_000.0
    north = R * math.radians(lat - START_LAT)
    east  = R * math.radians(lon - START_LON) * math.cos(math.radians(START_LAT))
    return east, north


def _school(species: str, east: float, north: float, depth: float,
            radius: float, density: float, **kw) -> FishSchool:
    ac = SPECIES_ACOUSTICS[species]
    return FishSchool(east, north, depth, radius, density, species,
                      ts_lf=ac["ts_lf"], ts_hf=ac["ts_hf"],
                      n_fish=ac["n_fish"], shape=ac["shape"], **kw)


def _make_schools(route_enu: List[Tuple[float,float]],
                  floor: FloorModel) -> List[FishSchool]:
    def pos(i): return route_enu[min(i, len(route_enu)-1)]

    p60 = pos(60); p30 = pos(30); p5 = pos(5)
    mid = len(route_enu)//2; cx = sum(p[0] for p in route_enu[mid-5:mid+5])/10
    cy  = sum(p[1] for p in route_enu[mid-5:mid+5])/10

    return [
        _school("bass",  p60[0], p60[1], floor.depth_at(*p60)*0.55, 8.0, 0.9,
                amp_e=10.0, amp_n=8.0,  freq=0.048, phase=0.0),
        _school("trout", p30[0], p30[1], floor.depth_at(*p30)*0.60, 6.0, 0.8,
                amp_e=12.0, amp_n=10.0, freq=0.071, phase=1.1),
        _school("carp",  cx-10,  cy-8,   6.5,                       10.0, 0.55,
                amp_e=6.0,  amp_n=5.0,  freq=0.031, phase=2.3),
        _school("bream", p5[0],  p5[1],  floor.depth_at(*p5)*0.65,  5.0, 0.8,
                amp_e=8.0,  amp_n=9.0,  freq=0.059, phase=3.7),
    ]


# ── Echo synthesis ────────────────────────────────────────────────────────────

def _fish_offsets(school: FishSchool) -> List[Tuple[float, float, float]]:
    """
    Deterministic individual fish positions (offsets from school centre).
    Demersal species (disk shape) are spread wide and flat; pelagic (sphere)
    fill a sphere.  The same school always produces the same offsets.
    """
    rng = random.Random(hash(school.species) ^ int(school.east_m * 10)
                        ^ int(school.north_m * 10))
    r        = school.radius_m
    dz_scale = 0.25 if school.shape == "disk" else 1.0
    offsets: List[Tuple[float, float, float]] = []
    attempts = 0
    while len(offsets) < school.n_fish and attempts < school.n_fish * 30:
        attempts += 1
        de = rng.uniform(-r, r)
        dn = rng.uniform(-r, r)
        dd = rng.uniform(-r * dz_scale, r * dz_scale)
        if de*de + dn*dn + (dd / dz_scale)**2 <= r*r:
            offsets.append((de, dn, dd))
    return offsets


def _make_echo_dual(depth_m: float, schools: List[FishSchool],
                    boat_e: float, boat_n: float,
                    rng: random.Random = None) -> Tuple[bytes, bytes]:
    """
    Return (echo_hf, echo_lf) — 200 kHz and 83 kHz channels.

    Both channels share the same structure:
      · Background noise
      · Floor return with two-way depth attenuation (HF attenuates faster)
      · Second harmonic (double-bounce)
      · Individual fish returns at correct slant range → arch shape in waterfall

    The LF/HF amplitude ratio is the primary species discriminant:
      bass/trout (physoclistous swimbladder) → LF dominant  (ratio > 1)
      carp/bream (physostomous / no resonance) → HF dominant (ratio < 1)
    """
    rng  = rng or random.Random()
    n    = ECHO_SIZE

    def _channel(alpha: float, ts_attr: str, tan_half: float) -> bytes:
        echo  = bytearray(rng.randint(2, 12) for _ in range(n))
        sigma = max(3, int(n * 0.012))

        def gauss(centre: int, amp: int, width: int) -> None:
            for i in range(max(0, centre - width*4), min(n, centre + width*4 + 1)):
                v = int(amp * math.exp(-0.5 * ((i - centre) / width) ** 2))
                echo[i] = min(255, echo[i] + v)

        # Two-way attenuation to the floor
        floor_atten = math.exp(-2.0 * alpha * depth_m)
        fi = min(n - 1, max(0, int(depth_m / MAX_RANGE_M * n)))
        gauss(fi, int(215 * floor_atten), sigma)
        if fi * 2 < n:
            gauss(fi * 2, int(80 * floor_atten ** 2), sigma + 2)

        # Individual fish returns with frequency-specific beam width.
        # Wider beam (LF) sees more fish; narrower (HF) gives better
        # angular resolution.  Beam taper attenuates off-axis targets.
        for s in schools:
            ts = getattr(s, ts_attr, 0.70)
            for de, dn, dd in _fish_offsets(s):
                fe = s.east_m  + de
                fn = s.north_m + dn
                fd = max(0.5, s.depth_m + dd)

                horiz = math.sqrt((fe - boat_e) ** 2 + (fn - boat_n) ** 2)
                beam_radius = fd * tan_half
                if horiz > beam_radius:
                    continue

                # Gaussian beam taper: targets at edge of beam return less signal
                beam_taper  = math.exp(-0.5 * (horiz / max(beam_radius, 1e-6)) ** 2)
                slant       = math.sqrt(horiz ** 2 + fd ** 2)
                fish_atten  = math.exp(-2.0 * alpha * fd)
                amp         = int(s.density * ts * fish_atten * beam_taper * 210)
                if amp < 1:
                    continue
                fi2 = min(n - 1, max(0, int(slant / MAX_RANGE_M * n)))
                if fi2 < fi - sigma * 2:
                    gauss(fi2, amp, 2)

        return bytes(echo)

    echo_hf = _channel(ALPHA_HF, "ts_hf", _TAN_HALF_HF)
    echo_lf = _channel(ALPHA_LF, "ts_lf", _TAN_HALF_LF)
    return echo_hf, echo_lf


# ── Boat physics ──────────────────────────────────────────────────────────────

class _Boat:
    def __init__(self, floor: FloorModel):
        self.floor   = floor
        self.lat     = START_LAT
        self.lon     = START_LON
        self.heading = START_HEADING
        self.speed   = SPEED_KTS
        self.t       = 0.0

    def step(self, dt: float):
        self.t += dt
        # Constant ~3°/s turn closes a full circle in 120s at 3.5kts (radius ~34m).
        # Small sinusoidal variation adds realism without breaking the closure.
        # A closed loop means depth at t=0 ≈ depth at t=120 — no systematic trend.
        turn_rate = 3.0 + 1.5 * math.sin(self.t * 0.4)
        self.heading = (self.heading + turn_rate * dt) % 360.0
        self.speed = max(0.5, SPEED_KTS + 0.3 * math.sin(self.t * 0.11))
        speed_ms = self.speed * 0.5144
        h = math.radians(self.heading)
        self.lat += speed_ms*dt*math.cos(h)/111_320.0
        self.lon += speed_ms*dt*math.sin(h)/(111_320.0*math.cos(math.radians(self.lat)))

    def depth(self) -> float:
        e, n = _enu(self.lat, self.lon)
        return max(0.6, self.floor.depth_at(e, n) + 0.04*random.gauss(0,1))

    def strength(self) -> int:
        return max(100, min(255, 185 - int(self.speed*8) + random.randint(-12,12)))


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class GeneratedSession:
    ticks:        List[Tick]
    fish_schools: List[FishSchool]
    floor:        FloorModel
    start_ts:     float
    duration_s:   float

    def to_ground_truth(self) -> dict:
        return {
            "origin_lat":  START_LAT,
            "origin_lon":  START_LON,
            "fish_schools": [
                {
                    "east_m":   round(s.east_m, 3),
                    "north_m":  round(s.north_m, 3),
                    "depth_m":  round(s.depth_m, 3),
                    "radius_m": s.radius_m,
                    "density":  s.density,
                    "species":  SPECIES_NAMES.get(s.species, s.species),
                    "amp_e":    s.amp_e,
                    "amp_n":    s.amp_n,
                    "freq":     s.freq,
                    "phase":    s.phase,
                    "ts_lf":    s.ts_lf,
                    "ts_hf":    s.ts_hf,
                    "lf_hf_ratio": round(s.ts_lf / s.ts_hf, 3),
                    "n_fish":   s.n_fish,
                    "shape":    s.shape,
                }
                for s in self.fish_schools
            ],
            "floor_grid": self.floor.sample_grid(step=1),  # 2m/cell — matches FloorModel resolution
        }


def generate(duration_s: float = 120.0,
             session_start_ts: float = 0.0,
             seed: int = 42) -> GeneratedSession:
    """
    Generate a synthetic fishing session.

    Returns a GeneratedSession with:
      - ticks: ordered list of Tick objects (deterministic timestamps)
      - fish_schools: ground truth fish positions
      - floor: the FloorModel used
    """
    random.seed(seed)
    floor  = FloorModel(seed=seed)
    boat   = _Boat(floor)
    ticks: List[Tick] = []
    route_enu: List[Tuple[float,float]] = []

    sonar_dt  = 1.0 / SONAR_HZ
    gps_every = SONAR_HZ // GPS_HZ
    n_steps   = int(duration_s * SONAR_HZ)

    for step in range(n_steps):
        boat.step(sonar_dt)
        ts_sonar = session_start_ts + step * sonar_dt
        e, n = _enu(boat.lat, boat.lon)
        depth = boat.depth()

        sonar = SonarTick(
            ts        = ts_sonar,
            depth_m   = depth,
            temp_c    = WATER_TEMP_C,
            signal_db = boat.strength() / 255.0 * 100.0,
            echo      = b"",   # filled after schools are placed
        )

        gps_tick = None
        if step % gps_every == 0:
            ts_gps = session_start_ts + (step // gps_every) * (1.0 / GPS_HZ)
            gps_tick = GpsTick(
                ts          = ts_gps,
                lat         = boat.lat,
                lon         = boat.lon,
                speed_kts   = boat.speed,
                heading_deg = boat.heading,
                hdop        = round(0.8 + 0.35*abs(math.sin(boat.t*0.031)), 1),
            )
            route_enu.append((e, n))

        ticks.append(Tick(ts=ts_sonar, sonar=sonar, gps=gps_tick))

    # Place fish schools at known route positions, then re-encode echo data
    schools = _make_schools(route_enu, floor)
    from synthetic.forward_scan import generate as _gen_fwd
    final_ticks: List[Tick] = []
    for i, (tick, step) in enumerate(zip(ticks, range(n_steps))):
        sonar = tick.sonar
        if sonar is not None:
            idx = i // gps_every
            if idx < len(route_enu):
                be, bn = route_enu[idx]
            else:
                be, bn = 0.0, 0.0
            # Compute school positions at this tick's time
            t_rel = tick.ts - session_start_ts
            schools_now = [s.at(t_rel) for s in schools]
            # Deterministic per-tick rngs derived from the session seed
            echo_rng = random.Random(seed ^ (i * 7) & 0xFFFFFF)
            fwd = None
            if tick.gps is not None:
                fwd_rng = random.Random(seed ^ (i * 3 + 1) & 0xFFFFFF)
                fwd = _gen_fwd(be, bn, tick.gps.heading_deg, floor, schools_now, fwd_rng)
            echo_hf, echo_lf = _make_echo_dual(sonar.depth_m, schools_now, be, bn, echo_rng)
            new_sonar = SonarTick(
                ts=sonar.ts, depth_m=sonar.depth_m, temp_c=sonar.temp_c,
                signal_db=sonar.signal_db,
                echo=echo_hf,
                echo_lf=echo_lf,
                forward_scan=fwd,
            )
            final_ticks.append(Tick(ts=tick.ts, sonar=new_sonar, gps=tick.gps))
        else:
            final_ticks.append(tick)

    return GeneratedSession(
        ticks        = final_ticks,
        fish_schools = schools,
        floor        = floor,
        start_ts     = session_start_ts,
        duration_s   = duration_s,
    )
