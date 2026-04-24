"""
src/synthetic/floor.py

Procedural 2D bathymetric height map using fractal Brownian motion
(fBm) and domain warping — the same technique used in game terrain
generators (Inigo Quilez style).

No external dependencies beyond numpy.
"""
import numpy as np
from typing import Optional


class FloorModel:
    GRID_SIZE_M = 500
    CELL_SIZE_M = 2.0
    MIN_DEPTH_M = 10.9
    MAX_DEPTH_M = 32.0

    def __init__(self, seed: int = 1337):
        n = int(self.GRID_SIZE_M / self.CELL_SIZE_M)
        self._n    = n
        self._half = self.GRID_SIZE_M / 2.0

        xs = np.linspace(-self._half, self._half, n, dtype=np.float32)
        ys = np.linspace(-self._half, self._half, n, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")

        def _fbm(Px, Py, r, octaves=7, lac=2.0, pers=0.45, freq=0.008):
            out, amp, f = np.zeros_like(Px), 1.0, freq
            for _ in range(octaves):
                a = r.uniform(0, 2 * np.pi); ph = r.uniform(0, 2 * np.pi)
                out += amp * np.sin(f * (np.cos(a)*Px + np.sin(a)*Py) + ph)
                amp *= pers; f *= lac
            return (out / sum(pers**i for i in range(octaves))).astype(np.float32)

        def _ridged(Px, Py, r, octaves=5, lac=2.1, pers=0.5, freq=0.012):
            out, amp, f = np.zeros_like(Px), 1.0, freq
            for _ in range(octaves):
                a = r.uniform(0, 2*np.pi); ph = r.uniform(0, 2*np.pi)
                out += amp * (1.0 - np.abs(np.sin(f*(np.cos(a)*Px+np.sin(a)*Py)+ph)))
                amp *= pers; f *= lac
            return (out / sum(pers**i for i in range(octaves))).astype(np.float32)

        rs = [np.random.RandomState(seed + i) for i in range(10)]
        # Domain warp level 1
        wx = 70.0 * _fbm(XX+3.7, YY+17.4, rs[0], octaves=5, freq=0.006)
        wy = 70.0 * _fbm(XX+14.1,YY+5.9,  rs[1], octaves=5, freq=0.006)
        WX1, WY1 = XX + wx, YY + wy
        # Domain warp level 2
        WX2 = WX1 + 40.0*_fbm(WX1+8.3, WY1+2.1, rs[2], octaves=4, freq=0.010)
        WY2 = WY1 + 40.0*_fbm(WX1+1.6, WY1+9.7, rs[3], octaves=4, freq=0.010)

        terrain = _fbm(WX2, WY2, rs[4], octaves=8, freq=0.007, pers=0.48)
        dr = self.MAX_DEPTH_M - self.MIN_DEPTH_M
        grid = (self.MIN_DEPTH_M + (terrain*0.5+0.5)*dr).astype(np.float32)
        grid += (_ridged(WX1, WY1, rs[5]) - 0.5) * 4.0

        # Channel
        ch_cx = 0.15*XX + 28.0*np.sin(XX/100.0) + 12.0*np.sin(XX/40.0)
        grid += 5.0 * np.exp(-0.5*((YY - ch_cx)/35.0)**2)

        # Shore falloff
        edge = np.minimum(np.minimum(XX+self._half, self._half-XX),
                          np.minimum(YY+self._half, self._half-YY))
        sf = np.clip(edge/60.0, 0.0, 1.0)
        grid = grid*sf + 1.5*(1.0-sf)

        self._grid = np.clip(grid, self.MIN_DEPTH_M, self.MAX_DEPTH_M)

    def depth_at(self, east_m: float, north_m: float) -> float:
        ix = int((east_m  + self._half) / self.CELL_SIZE_M)
        iy = int((north_m + self._half) / self.CELL_SIZE_M)
        return float(self._grid[
            max(0, min(self._n-1, ix)),
            max(0, min(self._n-1, iy)),
        ])

    def sample_grid(self, step: int = 5) -> dict:
        coarse = self._grid[::step, ::step].tolist()
        return {
            "rows":           len(coarse),
            "cols":           len(coarse[0]) if coarse else 0,
            "cell_size_m":    self.CELL_SIZE_M * step,
            "origin_east_m":  -self._half,
            "origin_north_m": -self._half,
            "depth_m":        [[round(v, 2) for v in row] for row in coarse],
        }
