"""
src/ticks/live.py

Reads NMEA sentences from a serial port in real-time.
Yields GpsTick objects as they arrive.
Uses actual NMEA timestamps — not time.time() — so recordings can be
replayed deterministically.
"""
import asyncio
from typing import AsyncIterator, Optional
from ticks.models import GpsTick
from ticks.nmea_decoder import parse_line

class LiveNMEAReader:
    """
    Async generator that connects to a serial port and yields GpsTicks.

    Usage:
        reader = LiveNMEAReader("/dev/ttyUSB0", baud=38400)
        async for tick in reader.stream():
            print(tick)
    """

    BAUD = 38400

    def __init__(self, port: str, baud: int = 38400):
        self.port = port
        self.baud = baud

    async def stream(self) -> AsyncIterator[GpsTick]:
        import serial_asyncio  # type: ignore
        reader, _ = await serial_asyncio.open_serial_connection(
            url=self.port, baudrate=self.baud)
        print(f"[LiveNMEAReader] Connected: {self.port} @ {self.baud}")
        while True:
            try:
                raw  = await reader.readline()
                line = raw.decode("ascii", errors="ignore").strip()
                tick = parse_line(line)
                if tick is not None:
                    yield tick
            except Exception as exc:
                print(f"[LiveNMEAReader] Error: {exc}")
