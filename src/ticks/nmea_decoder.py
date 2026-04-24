"""
src/ticks/nmea_decoder.py

Parses NMEA 0183 sentences → GpsTick.
Timestamp comes from the HHMMSS field in the sentence — deterministic.
"""
import re
from typing import Optional, List, Iterator
from ticks.models import GpsTick

def _nmea_checksum_valid(line: str) -> bool:
    if "*" not in line:
        return False
    body, cs = line[1:].rsplit("*", 1)
    expected = 0
    for ch in body:
        expected ^= ord(ch)
    return cs.strip().upper() == f"{expected:02X}"

def _nmea_ts(hhmmss: str, date_offset: float = 0.0) -> float:
    """Convert HHMMSS.ss string to seconds-since-midnight + date_offset."""
    try:
        h  = int(hhmmss[0:2])
        m  = int(hhmmss[2:4])
        s  = float(hhmmss[4:])
        return date_offset + h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return date_offset

def _dec_to_dd(nmea_val: str, hemi: str) -> float:
    """NMEA ddmm.mmmmm → decimal degrees."""
    if not nmea_val:
        return 0.0
    dot = nmea_val.index(".")
    deg = int(nmea_val[:dot - 2])
    mins = float(nmea_val[dot - 2:])
    dd = deg + mins / 60.0
    return -dd if hemi in ("S", "W") else dd

def parse_line(line: str, date_offset: float = 0.0) -> Optional[GpsTick]:
    """Parse one NMEA sentence. Returns GpsTick or None."""
    line = line.strip()
    if not line.startswith("$"):
        return None
    if not _nmea_checksum_valid(line):
        return None

    parts = line[1:].split("*")[0].split(",")
    sentence = parts[0]

    try:
        if sentence in ("GPGGA", "GNGGA"):
            ts  = _nmea_ts(parts[1], date_offset)
            lat = _dec_to_dd(parts[2], parts[3])
            lon = _dec_to_dd(parts[4], parts[5])
            if lat == 0.0 and lon == 0.0:
                return None
            hdop = float(parts[8]) if parts[8] else 1.0
            return GpsTick(ts=ts, lat=lat, lon=lon,
                           speed_kts=0.0, heading_deg=0.0, hdop=hdop)

        if sentence in ("GPRMC", "GNRMC"):
            ts  = _nmea_ts(parts[1], date_offset)
            if parts[2] != "A":           # not active
                return None
            lat = _dec_to_dd(parts[3], parts[4])
            lon = _dec_to_dd(parts[5], parts[6])
            spd = float(parts[7]) if parts[7] else 0.0
            hdg = float(parts[8]) if parts[8] else 0.0
            return GpsTick(ts=ts, lat=lat, lon=lon,
                           speed_kts=spd, heading_deg=hdg)
    except (ValueError, IndexError):
        pass
    return None

def parse_file(path: str, date_offset: float = 0.0) -> List[GpsTick]:
    """Parse an NMEA file and return all valid GpsTicks, sorted by ts."""
    ticks: List[GpsTick] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            t = parse_line(line, date_offset)
            if t is not None:
                ticks.append(t)
    return sorted(ticks, key=lambda t: t.ts)

def iter_file(path: str, date_offset: float = 0.0) -> Iterator[GpsTick]:
    with open(path, "r", errors="ignore") as f:
        for line in f:
            t = parse_line(line, date_offset)
            if t is not None:
                yield t
