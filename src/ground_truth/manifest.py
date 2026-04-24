"""
src/ground_truth/manifest.py
"""
from __future__ import annotations
import json
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GroundTruth:
    origin_lat:   float
    origin_lon:   float
    fish_schools: List[dict]
    floor_grid:   Optional[dict] = None

    @classmethod
    def from_dict(cls, d: dict) -> "GroundTruth":
        known = {"origin_lat","origin_lon","fish_schools","floor_grid"}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_file(cls, path: str) -> "GroundTruth":
        return cls.from_dict(json.loads(pathlib.Path(path).read_text()))

    def to_file(self, path: str) -> None:
        pathlib.Path(path).write_text(json.dumps({
            "origin_lat":   self.origin_lat,
            "origin_lon":   self.origin_lon,
            "fish_schools": self.fish_schools,
            "floor_grid":   self.floor_grid,
        }, indent=2))
