"""
src/ticks/recorder.py

Records Ticks to a SQLite .ticks file.

Schema:
    ticks(id INTEGER PK, ts REAL, kind TEXT, payload TEXT, blob BLOB)
    session(key TEXT, value TEXT)

kind: 'sonar' | 'gps'
payload: JSON-encoded dict of the tick's scalar fields.
blob:    raw echo bytes for sonar; for gps rows, NULL.
         For sonar rows, an additional 'fwd' BLOB row is written separately
         (kind='fwd_scan') so the schema stays simple.
"""
import sqlite3
import json
import os
from typing import Optional
from ticks.models import Tick, SonarTick, GpsTick

SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL    NOT NULL,
    kind    TEXT    NOT NULL,
    payload TEXT    NOT NULL,
    blob    BLOB
);
CREATE INDEX IF NOT EXISTS ticks_ts ON ticks(ts);
CREATE TABLE IF NOT EXISTS session (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
"""

class Recorder:
    def __init__(self, path: str):
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._buf: list = []
        self._buf_size = 100   # flush every N ticks

    def record(self, tick: Tick) -> None:
        if tick.sonar:
            s = tick.sonar
            # Pack echo + forward_scan into one BLOB:
            # [u32 echo_len][echo bytes][u32 fwd_len][fwd bytes]
            import struct as _struct
            echo = s.echo or b""
            fwd  = s.forward_scan or b""
            blob = (_struct.pack("<I", len(echo)) + echo
                    + _struct.pack("<I", len(fwd)) + fwd) if (echo or fwd) else None
            self._buf.append((s.ts, "sonar", json.dumps({
                "ts": s.ts, "depth_m": s.depth_m,
                "temp_c": s.temp_c, "signal_db": s.signal_db,
            }), blob))
        if tick.gps:
            g = tick.gps
            self._buf.append((g.ts, "gps", json.dumps({
                "ts": g.ts, "lat": g.lat, "lon": g.lon,
                "speed_kts": g.speed_kts, "heading_deg": g.heading_deg,
                "hdop": g.hdop,
            }), None))
        if len(self._buf) >= self._buf_size:
            self.flush()

    def set_metadata(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO session(key, value) VALUES (?,?)",
            (key, value))

    def flush(self) -> None:
        if self._buf:
            self._conn.executemany(
                "INSERT INTO ticks(ts, kind, payload, blob) VALUES (?,?,?,?)",
                self._buf)
            self._conn.commit()
            self._buf.clear()

    def close(self) -> None:
        self.flush()
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
