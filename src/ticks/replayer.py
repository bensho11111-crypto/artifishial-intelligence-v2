"""
src/ticks/replayer.py

Replays ticks from a .ticks SQLite recording.

state_at(ts) uses the SQL index to fetch only rows WHERE ts <= ts,
giving O(log N) random access to any point in the recording.
"""
import sqlite3
import json
from typing import Iterator, List, Optional, Tuple
from ticks.models import Tick, SonarTick, GpsTick

def _row_to_tick(row: Tuple) -> Optional[Tick]:
    # Row layout: id, ts, kind, payload, blob (blob may be absent in legacy DBs)
    if len(row) >= 5:
        _, ts, kind, payload_str, blob = row
    else:
        _, ts, kind, payload_str = row
        blob = None
    try:
        d = json.loads(payload_str)
    except (json.JSONDecodeError, TypeError):
        return None

    if kind == "sonar":
        echo = b""
        fwd  = None
        if blob:
            import struct as _struct
            buf = bytes(blob)
            n_echo = _struct.unpack_from("<I", buf, 0)[0]
            echo = buf[4:4 + n_echo]
            off  = 4 + n_echo
            if off + 4 <= len(buf):
                n_fwd = _struct.unpack_from("<I", buf, off)[0]
                fwd_bytes = buf[off + 4:off + 4 + n_fwd]
                fwd = fwd_bytes if n_fwd > 0 else None
        return Tick(ts=ts, sonar=SonarTick(
            ts=d["ts"], depth_m=d["depth_m"],
            temp_c=d["temp_c"], signal_db=d["signal_db"],
            echo=echo, forward_scan=fwd,
        ))
    if kind == "gps":
        return Tick(ts=ts, gps=GpsTick(
            ts=d["ts"], lat=d["lat"], lon=d["lon"],
            speed_kts=d["speed_kts"], heading_deg=d["heading_deg"],
            hdop=d.get("hdop", 1.0),
        ))
    return None

class Replayer:
    """
    Provides seekable, ordered access to a .ticks recording.

    Usage:
        r = Replayer("session.ticks")
        # iterate all ticks in order
        for tick in r.iter_all():
            ...
        # get only ticks up to ts=T (for backtest)
        for tick in r.iter_up_to(T):
            ...
        # duration / metadata
        print(r.duration_s)
    """

    def __init__(self, path: str):
        self._conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def iter_all(self) -> Iterator[Tick]:
        cur = self._conn.execute(
            "SELECT id, ts, kind, payload, blob FROM ticks ORDER BY ts ASC")
        for row in cur:
            t = _row_to_tick(row)
            if t is not None:
                yield t

    def iter_up_to(self, ts: float) -> Iterator[Tick]:
        """Yield all ticks with ts <= ts, in order. O(log N) seek via index."""
        cur = self._conn.execute(
            "SELECT id, ts, kind, payload, blob FROM ticks "
            "WHERE ts <= ? ORDER BY ts ASC", (ts,))
        for row in cur:
            t = _row_to_tick(row)
            if t is not None:
                yield t

    def iter_range(self, t_start: float, t_end: float) -> Iterator[Tick]:
        cur = self._conn.execute(
            "SELECT id, ts, kind, payload, blob FROM ticks "
            "WHERE ts >= ? AND ts <= ? ORDER BY ts ASC", (t_start, t_end))
        for row in cur:
            t = _row_to_tick(row)
            if t is not None:
                yield t

    @property
    def start_ts(self) -> Optional[float]:
        row = self._conn.execute("SELECT MIN(ts) FROM ticks").fetchone()
        return row[0] if row else None

    @property
    def end_ts(self) -> Optional[float]:
        row = self._conn.execute("SELECT MAX(ts) FROM ticks").fetchone()
        return row[0] if row else None

    @property
    def duration_s(self) -> float:
        s, e = self.start_ts, self.end_ts
        return (e - s) if (s is not None and e is not None) else 0.0

    def get_metadata(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM session WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
