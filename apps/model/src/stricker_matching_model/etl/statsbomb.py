"""StatsBomb ETL focused on producing model-ready rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Generator, Iterable

import pandas as pd
from tqdm import tqdm

from stricker_matching_model.etl.base import BaseETL
from stricker_matching_model.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StatsBombETL(BaseETL):
    """ETL for model training data.

    This class is intentionally scoped to preparing rows for the model pipeline.
    Reference-table generation lives in etl/reference_tables.py.
    """

    data_path: Path | None = None
    output_path: Path | None = None
    output_format: str = "json"
    flip_left_to_right: bool = False
    pitch_length: float = 120.0
    pitch_width: float = 80.0
    target_length: float | None = 105.0
    target_width: float | None = 68.0

    def extract(self) -> Iterable[Path]:
        """Iterate over raw StatsBomb event files."""
        if self.data_path is None:
            raise FileNotFoundError("data_path is required to extract StatsBomb data")
        events_root = self.data_path / "raw" / "events"
        if not events_root.exists():
            raise FileNotFoundError(f"Events directory not found at {events_root}")
        files = sorted(events_root.glob("*.json"))
        logger.info("Found %s event files in %s", len(files), events_root)
        return files

    def transform(
        self, raw: Iterable[Path]
    ) -> Generator[tuple[int, pd.DataFrame], None, None]:
        """Load and normalize one match at a time.

        Yields (match_id, normalized_events) for streaming processing.
        """
        logger.info("Normalizing event files")
        for file_path in tqdm(raw, desc="Normalize events"):
            if logger.isEnabledFor(logging.DEBUG):
                tqdm.write(f"Normalizing match {file_path.stem}")
            match_id = int(file_path.stem)
            events = self.load_events(file_path)
            normalized = self.normalize_coords(events)
            yield match_id, normalized

    def load(self, transformed: Iterable[tuple[int, pd.DataFrame]]) -> list[Path]:
        """Persist normalized events per match and return output paths."""
        output_root = self._resolve_output_path()
        output_root.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Writing normalized events to %s as %s",
            output_root,
            self.output_format,
        )

        written: list[Path] = []
        for match_id, events in tqdm(transformed, desc="Write normalized"):
            if logger.isEnabledFor(logging.DEBUG):
                tqdm.write(f"Writing normalized events for match {match_id}")
            path = self.write_normalized(match_id, events, output_root)
            written.append(path)
        logger.info("Wrote %s normalized event files", len(written))
        return written

    def load_events(self, file_path: Path) -> pd.DataFrame:
        events = pd.read_json(file_path)
        events = events.dropna(
            subset=["location"]
        ).copy()  # location only exists if the event has pitch coordinates
        events[["x", "y"]] = pd.DataFrame(
            events["location"].tolist(), index=events.index
        )
        return events

    def normalize_coords(self, events: pd.DataFrame) -> pd.DataFrame:
        out = events.copy()
        x = out["x"] / self.pitch_length
        y = out["y"] / self.pitch_width

        if self.flip_left_to_right:
            x = 1.0 - x
            y = 1.0 - y

        if self.target_length is not None and self.target_width is not None:
            x = x * self.target_length
            y = y * self.target_width

        out["x_norm"] = x
        out["y_norm"] = y
        return out

    def write_normalized(
        self, match_id: int, events: pd.DataFrame, output_root: Path
    ) -> Path:
        suffix = "json" if self.output_format == "json" else "csv"
        output_file = output_root / f"{match_id}.{suffix}"

        if self.output_format == "json":
            events.to_json(output_file, index=False)
        elif self.output_format == "csv":
            events.to_csv(output_file, index=False)
        else:
            raise ValueError(
                f"Unsupported output_format: {self.output_format}. Use json or csv."
            )

        return output_file

    def _resolve_output_path(self) -> Path:
        if self.output_path:
            return self.output_path
        if not self.data_path:
            raise FileNotFoundError("output_path or data_path is required")
        return self.data_path / "processed" / "events"
