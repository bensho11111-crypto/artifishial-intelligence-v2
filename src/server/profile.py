"""
src/server/profile.py

Per-second frame profiler for the WebSocket loop.

Captures timings for each logical stage in the path between two replay
seconds advancing, then emits a one-line-per-stage summary when the
replay clock crosses an integer-second boundary.

Usage:
    prof = FrameProfiler(enabled=True)
    while True:
        prof.start_frame(replay_ts=ts)        # marks t0 + decides whether to flush
        ... work A ...; prof.stage("A")
        ... work B ...; prof.stage("B")
        prof.end_frame(payload_bytes=len(b))

Stages are accumulated into the current bucket (one bucket per integer
replay-second). When start_frame sees the second has advanced,
the previous bucket is flushed to stdout and a fresh one begins.

Disabled mode (default): every method is a near-zero-cost no-op so the
profiler can stay wired in production without measurable overhead.
"""
from __future__ import annotations

import os
import statistics
import time
from typing import Optional


class FrameProfiler:
    # Width-padded column order — appears in this order in the output.
    DEFAULT_STAGES = (
        "state_at",
        "pointcloud_build",
        "extract_boat",
        "echo_lookup",
        "fwd_scan_lookup",
        "ground_truth",
        "mesh_check",
        "mesh_build",
        "json_encode",
        "ws_send",
        "sleep",
    )

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._cur_second: int = -1
        self._buckets: dict[str, list[float]] = {}
        self._frames: int = 0
        self._bytes: int = 0
        self._mesh_built: int = 0      # how many times mesh was rebuilt this bucket
        self._t_frame_start: float = 0.0
        self._t_stage_start: float = 0.0

    # ── Frame lifecycle ──────────────────────────────────────────────────────

    def start_frame(self, replay_ts: Optional[float]) -> None:
        if not self.enabled:
            return
        sec = -1 if replay_ts is None else int(replay_ts)
        if sec != self._cur_second:
            if self._cur_second >= 0 and self._frames > 0:
                self._emit()
            self._reset(sec)
        now = time.perf_counter()
        self._t_frame_start = now
        self._t_stage_start = now

    def stage(self, name: str) -> None:
        """Record elapsed time since the previous mark under `name`."""
        if not self.enabled:
            return
        now = time.perf_counter()
        self._buckets.setdefault(name, []).append((now - self._t_stage_start) * 1000.0)
        self._t_stage_start = now

    def note_mesh_built(self) -> None:
        if self.enabled:
            self._mesh_built += 1

    def end_frame(self, payload_bytes: int = 0) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._buckets.setdefault("TOTAL", []).append((now - self._t_frame_start) * 1000.0)
        self._frames += 1
        self._bytes += payload_bytes

    # ── Internal ─────────────────────────────────────────────────────────────

    def _reset(self, sec: int) -> None:
        self._cur_second = sec
        self._buckets = {}
        self._frames = 0
        self._bytes = 0
        self._mesh_built = 0

    def _emit(self) -> None:
        sec = self._cur_second
        n   = self._frames
        kib = self._bytes / 1024.0

        ordered = [s for s in self.DEFAULT_STAGES if s in self._buckets]
        # Surface any unknown stage names at the end so nothing is silently dropped.
        ordered += [s for s in self._buckets if s not in self.DEFAULT_STAGES and s != "TOTAL"]
        if "TOTAL" in self._buckets:
            ordered.append("TOTAL")

        header = (f"[profile s={sec:4d}]  frames={n:3d}  "
                  f"bytes={kib:8.1f} KiB  mesh_rebuilds={self._mesh_built}")
        lines = [header,
                 f"    {'stage':>18}  {'mean':>7}  {'p50':>7}  {'max':>7}  {'count':>5}"]
        for name in ordered:
            samples = self._buckets[name]
            if not samples:
                continue
            mean   = statistics.mean(samples)
            median = statistics.median(samples)
            mx     = max(samples)
            lines.append(f"    {name:>18}  {mean:7.2f}  {median:7.2f}  "
                         f"{mx:7.2f}  {len(samples):5d}")
        # Single print so concurrent loop output stays grouped.
        print("\n".join(lines), flush=True)


def profiler_from_env() -> FrameProfiler:
    """Construct a profiler enabled iff the WS_PROFILE env var is truthy."""
    raw = os.environ.get("WS_PROFILE", "").strip().lower()
    return FrameProfiler(enabled=raw in ("1", "true", "yes", "on"))
