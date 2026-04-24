"""
src/ticks/models.py

Core immutable data types.  Every value here comes from hardware or a
recording file — never from wall-clock time — so backtests are fully
deterministic and reproducible.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SonarTick:
    """One sonar ping from the transducer (Lowrance SL2/SL3 packet)."""
    ts: float           # seconds — from hardware packet header, NOT time.time()
    depth_m: float
    temp_c: float
    signal_db: float    # 0–100
    echo: bytes         # raw A-scope amplitude array (512 bytes)


@dataclass(frozen=True)
class GpsTick:
    """One GPS fix from the NMEA stream."""
    ts: float           # seconds — from NMEA sentence timestamp
    lat: float          # decimal degrees
    lon: float          # decimal degrees
    speed_kts: float
    heading_deg: float
    hdop: float = 1.0


@dataclass(frozen=True)
class Tick:
    """
    A single timestamped event from the data stream.
    Either sonar, gps, or both may be present depending on what the
    hardware sent at this moment.
    """
    ts: float
    sonar: Optional[SonarTick] = None
    gps:   Optional[GpsTick]   = None

    def has_sonar(self) -> bool:
        return self.sonar is not None

    def has_gps(self) -> bool:
        return self.gps is not None


@dataclass(frozen=True)
class Observation:
    """
    A single georeferenced depth measurement ready for WorldState.
    Produced by processing/fusion.py from one or more Ticks.
    """
    ts: float
    east_m: float       # ENU metres relative to session origin
    north_m: float
    depth_m: float
    confidence: float   # 0.0–1.0
    heading_deg: float
    speed_kts: float
