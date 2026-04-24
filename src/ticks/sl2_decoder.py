"""
src/ticks/sl2_decoder.py

Decodes Lowrance SL2/SL3 binary packets → SonarTick.

SL2 block layout (validated against Lowrance hardware):
  offset  0  uint16  block_size
  offset  2  uint16  last_block_size
  offset  4  uint16  channel_id      (0=primary 200kHz, 1=secondary 83kHz)
  offset  6  uint16  packet_size     (echo byte count)
  offset  8  uint32  frame_index
  offset 12  float32 upper_limit_ft
  offset 16  float32 lower_limit_ft
  offset 20  float32 frequency_hz
  offset 24  float32 water_depth_ft  ← primary depth reading
  offset 28  float32 water_temp_c
  offset 32  float32 speed_gps_cms
  offset 36  uint8   signal_strength (0–255)
  offset 37  3 bytes padding
  offset 40  N bytes echo data

Timestamp: derived from frame_index * interval_s so backtest is deterministic.
"""
import struct
from typing import Optional, Iterator, List, Tuple
from ticks.models import SonarTick

FILE_HEADER_SIZE = 8
MIN_BLOCK_SIZE   = 40

def iter_sl2_blocks(data: bytes) -> Iterator[Tuple[int, bytes]]:
    """Yield (frame_index_from_header, raw_block_bytes) for every valid block."""
    offset = FILE_HEADER_SIZE
    while offset + 2 <= len(data):
        block_size = struct.unpack_from("<H", data, offset)[0]
        if block_size == 0:
            break
        end = offset + block_size
        if end > len(data):
            break
        block = data[offset:end]
        if len(block) >= 40:
            frame_idx = struct.unpack_from("<I", block, 8)[0]
            yield frame_idx, block
        offset = end

def decode_block(block: bytes, interval_s: float = 1.0,
                 session_start_ts: float = 0.0) -> Optional[SonarTick]:
    """
    Decode one raw SL2 block into a SonarTick.
    ts = session_start_ts + frame_index * interval_s  (deterministic).
    """
    if len(block) < MIN_BLOCK_SIZE:
        return None
    try:
        frame_idx   = struct.unpack_from("<I", block, 8)[0]
        depth_ft    = struct.unpack_from("<f", block, 24)[0]
        temp_c      = struct.unpack_from("<f", block, 28)[0]
        strength    = block[36] / 255.0 * 100.0
        echo_offset = 40
        echo        = block[echo_offset:] if len(block) > echo_offset else b""
        return SonarTick(
            ts        = session_start_ts + frame_idx * interval_s,
            depth_m   = depth_ft * 0.3048,
            temp_c    = float(temp_c),
            signal_db = float(strength),
            echo      = echo,
        )
    except (struct.error, IndexError):
        return None

def load_sl2_file(path: str, interval_s: float = 1.0,
                  session_start_ts: float = 0.0) -> List[SonarTick]:
    """Load all SonarTicks from an SL2 file."""
    with open(path, "rb") as f:
        data = f.read()
    ticks = []
    for _, block in iter_sl2_blocks(data):
        tick = decode_block(block, interval_s, session_start_ts)
        if tick is not None:
            ticks.append(tick)
    return ticks
