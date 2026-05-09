"""
src/server/stream_controller.py

Coordinates a "restart" between the WS handler and the stream loop.

The stream loop awaits restart_event in its outer while-True; the WS handler
calls request_restart() to set it. The loop is responsible for resetting
fusion + world_state and re-iterating ticks from the start.
"""
import asyncio


class StreamController:
    def __init__(self):
        self._restart_event = asyncio.Event()

    def request_restart(self) -> None:
        self._restart_event.set()

    @property
    def restart_event(self) -> asyncio.Event:
        return self._restart_event

    def consume_restart(self) -> bool:
        """Return True if a restart was requested; clears the flag."""
        was_set = self._restart_event.is_set()
        self._restart_event.clear()
        return was_set

    def reset(self) -> None:
        """Reset the controller state (clear restart flag)."""
        self._restart_event.clear()
