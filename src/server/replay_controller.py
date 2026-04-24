"""
src/server/replay_controller.py

Tracks the current replay position and handles seek/pause/play commands
from the WebSocket client.
"""
import asyncio
from typing import Optional, Callable


class ReplayController:
    def __init__(self, duration_s: float, on_seek: Optional[Callable] = None):
        self._duration     = duration_s
        self._position     = 0.0
        self._paused       = False
        self._speed        = 1.0
        self._on_seek      = on_seek
        self._seek_event   = asyncio.Event()
        self._seek_target: Optional[float] = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def position_s(self) -> float:
        return self._position

    @property
    def duration_s(self) -> float:
        return self._duration

    @property
    def paused(self) -> bool:
        return self._paused

    # ── Commands ──────────────────────────────────────────────────────────────

    def pause(self) -> None:
        self._paused = True

    def play(self) -> None:
        # If the recording ended, restart from the beginning.
        if self._position >= self._duration:
            self._position = 0.0
        self._paused = False

    def set_speed(self, multiplier: float) -> None:
        self._speed = max(0.1, min(10.0, multiplier))

    async def seek(self, fraction: float) -> None:
        """Queue a seek to `fraction` (0.0–1.0) of the total duration."""
        self._seek_target = max(0.0, min(1.0, fraction)) * self._duration
        self._seek_event.set()

    # ── Tick ──────────────────────────────────────────────────────────────────

    def advance(self, dt: float) -> None:
        """
        Advance the position by `dt` seconds of wall-clock time, scaled by
        speed.  Apply any pending seek.  Call this from the replay loop once
        per iteration.
        """
        # Apply a pending seek first
        if self._seek_event.is_set():
            self._seek_event.clear()
            if self._seek_target is not None:
                self._position    = self._seek_target
                self._seek_target = None
                if self._on_seek:
                    self._on_seek()

        # Advance time if not paused; auto-pause at end so the user
        # can see the final state rather than silently looping.
        if not self._paused:
            self._position += dt * self._speed
            if self._position >= self._duration:
                self._position = self._duration
                self._paused = True   # let play() restart from 0
