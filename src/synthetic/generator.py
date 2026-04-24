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
_TAN_HALF     = math.tan(math.radians(10.0))


# ── Fish schools ──────────────────────────────────────────────────────────────

@dataclass
class FishSchool:
    east_m:   float
    north_m:  float
    depth_m:  float
    radius_m: float
    density:  float
    species:  str


SPECIES_TS = {"bass": 0.85, "trout": 0.70, "carp": 0.60, "bream": 0.50}
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


def _make_schools(route_enu: List[Tuple[float,float]],
                  floor: FloorModel) -> List[FishSchool]:
    def pos(i): return route_enu[min(i, len(route_enu)-1)]

    p60 = pos(60); p30 = pos(30); p5 = pos(5)
    mid = len(route_enu)//2; cx = sum(p[0] for p in route_enu[mid-5:mid+5])/10
    cy  = sum(p[1] for p in route_enu[mid-5:mid+5])/10

    return [
        FishSchool(p60[0], p60[1], floor.depth_at(*p60)*0.55, 8.0, 0.9, "bass"),
        FishSchool(p30[0], p30[1], floor.depth_at(*p30)*0.60, 6.0, 0.8, "trout"),
        FishSchool(cx-10, cy-8,    6.5,                       10.0, 0.55, "carp"),
        FishSchool(p5[0], p5[1],   floor.depth_at(*p5)*0.65,  5.0, 0.8, "bream"),
    ]


# ── Echo synthesis ────────────────────────────────────────────────────────────

def _make_echo(depth_m: float, schools: List[FishSchool],
               boat_e: float, boat_n: float) -> bytes:
    n = ECHO_SIZE
    echo = bytearray(n)
    rng  = random.Random()

    for i in range(n):
        echo[i] = rng.randint(2, 15)

    def gauss(centre, amp, width):
        for i in range(max(0, centre-width*4), min(n, centre+width*4+1)):
            v = int(amp * math.exp(-0.5*((i-centre)/width)**2))
            echo[i] = min(255, echo[i]+v)

    fi = min(n-1, max(0, int(depth_m/MAX_RANGE_M*n)))
    sigma = max(3, int(n*0.012))
    gauss(fi, 215, sigma)
    if fi*2 < n:
        gauss(fi*2, 80, sigma+2)

    for s in schools:
        horiz = math.sqrt((s.east_m-boat_e)**2 + (s.north_m-boat_n)**2)
        beam  = s.depth_m * _TAN_HALF
        if horiz > beam + s.radius_m:
            continue
        overlap = min(1.0, max(0.0, (beam+s.radius_m-horiz)/s.radius_m))
        slant = math.sqrt(horiz**2 + s.depth_m**2)
        amp = int(s.density * overlap * 0.7 * SPECIES_TS.get(s.species, 0.65) * 220)
        if amp < 1:
            continue
        fi2 = min(n-1, max(0, int(slant/MAX_RANGE_M*n)))
        if fi2 < fi - sigma*2:
            gauss(fi2, amp, 3)

    return bytes(echo)


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
    final_ticks: List[Tick] = []
    for i, (tick, step) in enumerate(zip(ticks, range(n_steps))):
        boat_step_e, boat_step_n = _enu(
            START_LAT, START_LON)   # approximate — echo only needs rough position
        # Reuse sonar with real echo
        sonar = tick.sonar
        if sonar is not None:
            idx = i // gps_every
            if idx < len(route_enu):
                be, bn = route_enu[idx]
            else:
                be, bn = 0.0, 0.0
            new_sonar = SonarTick(
                ts=sonar.ts, depth_m=sonar.depth_m, temp_c=sonar.temp_c,
                signal_db=sonar.signal_db,
                echo=_make_echo(sonar.depth_m, schools, be, bn),
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
