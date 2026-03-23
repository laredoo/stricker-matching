"""ETL utilities for StatsBomb data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class StatsBombETL:  # ToDo: not the best architecture for multiple providers, but good enough for first release.
    data_path: Path | None = None
    demo: bool = False

    def extract(self) -> list[list[float]]:
        if self.demo:
            return [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        if self.data_path is None:
            raise FileNotFoundError("data_path is required when demo=False")
        raise NotImplementedError("StatsBomb JSON loading not implemented yet")

    def transform(self, raw: list[list[float]]) -> list[list[float]]:
        return raw

    def load(self, cleaned: list[list[float]]) -> list[list[float]]:
        return cleaned
