"""
src/ticks/recorder.py

Records Ticks to a SQLite .ticks file.

Schema:
    ticks(id INTEGER PK, ts REAL, kind TEXT, payload BLOB)
    session(key TEXT, value TEXT)

kind: 'sonar' | 'gps'
payload: msgpack-encoded dict of the tick's fields (no echo bytes — too large)
         For sonar: {ts, depth_m, temp_c, signal_db}
         For gps:   {ts, lat, lon, speed_kts, heading_deg, hdop}
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
    payload TEXT    NOT NULL
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
            self._buf.append((s.ts, "sonar", json.dumps({
                "ts": s.ts, "depth_m": s.depth_m,
                "temp_c": s.temp_c, "signal_db": s.signal_db,
            })))
        if tick.gps:
            g = tick.gps
            self._buf.append((g.ts, "gps", json.dumps({
                "ts": g.ts, "lat": g.lat, "lon": g.lon,
                "speed_kts": g.speed_kts, "heading_deg": g.heading_deg,
                "hdop": g.hdop,
            })))
        if len(self._buf) >= self._buf_size:
            self.flush()

    def set_metadata(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO session(key, value) VALUES (?,?)",
            (key, value))

    def flush(self) -> None:
        if self._buf:
            self._conn.executemany(
                "INSERT INTO ticks(ts, kind, payload) VALUES (?,?,?)",
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
