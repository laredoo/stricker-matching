"""StatsBomb ETL focused on producing model-ready rows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
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
    output_format: str = "parquet"
    flip_left_to_right: bool = False
    pitch_length: float = 120.0
    pitch_width: float = 80.0
    target_length: float | None = 105.0
    target_width: float | None = 68.0
    overwrite_files: bool = False
    striker_position_ids: set[int] = field(
        default_factory=lambda: {
            17,  # Right Wing
            21,  # Left Wing
            22,  # Right Center Forward
            23,  # Stricker / Center Forward
            24,  # Left Center Forward
            25,  # Secondary Striker
        }
    )

    def extract(self) -> Iterable[Path]:
        """Iterate over raw StatsBomb event files."""
        logger.info("Extracting raw event files from %s", self.data_path)

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
            events = self.load_events(file_path, match_id)
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

        match_written: list[Path] = []
        for match_id, events in tqdm(transformed, desc="Write normalized"):
            if logger.isEnabledFor(logging.DEBUG):
                tqdm.write(f"Writing normalized events for match {match_id}")
            match_path = self.write_normalized(match_id, events, output_root)
            match_written.append(match_path)
        return match_written

    def load_events(self, file_path: Path, match_id: int) -> pd.DataFrame:
        striker_ids = self.load_striker_ids(match_id)
        events = pd.read_json(file_path)
        events = events.dropna(
            subset=["location"]
        ).copy()  # location only exists if the event has pitch coordinates
        events[["x", "y"]] = pd.DataFrame(
            events["location"].tolist(), index=events.index
        )
        if not striker_ids:
            if logger.isEnabledFor(logging.INFO):
                tqdm.write(f"Match {match_id} has no striker ids; dropping all events")
            return events.iloc[0:0].copy()
        if "player" not in events.columns:
            if logger.isEnabledFor(logging.INFO):
                tqdm.write(
                    f"Match {match_id} missing player column; dropping all events"
                )
            return events.iloc[0:0].copy()

        player_ids = events["player"].str.get("id")
        mask = player_ids.isin(striker_ids)
        filtered = events.loc[mask].copy()
        filtered["player_id"] = player_ids[mask].values.astype(int)
        filtered = filtered.reset_index(drop=True)
        if logger.isEnabledFor(logging.DEBUG):
            tqdm.write(
                f"Match {match_id} filtered events from {len(events)} rows to {len(filtered)} rows",
            )
        return filtered

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

    def write_match(
        self, match_id: int, events: pd.DataFrame, output_root: Path, suffix: str
    ) -> Path:
        matches_root = output_root / "matches"
        matches_root.mkdir(parents=True, exist_ok=True)

        match_file = matches_root / f"{match_id}.{suffix}"

        if self.output_format == "json":
            if match_file.exists() and not self.overwrite_files:
                if logger.isEnabledFor(logging.DEBUG):
                    tqdm.write(
                        f"Overwrite Files flag set to False and Match file {match_file} already exists; skipping write"
                    )
                return match_file
            events.to_json(match_file, index=False)
        else:
            raise ValueError(
                f"Unsupported output_format: {self.output_format}. Use json."
            )

        return match_file

    def write_players(  # ToDo: Refactor this method
        self, match_id: int, events: pd.DataFrame, output_root: Path, suffix: str
    ) -> Path:
        players_root = output_root / "players"
        players_root.mkdir(parents=True, exist_ok=True)

        if "player_id" not in events.columns:
            raise ValueError(
                f"Missing player_id column in match {match_id}; it is required in events to write player files"
            )

        for player_id, player_events in events.groupby("player_id"):
            if logger.isEnabledFor(logging.DEBUG):
                tqdm.write(f"Writing player {player_id} events for match {match_id}")

            player_events = player_events.copy()
            player_events["match_id"] = match_id

            player_file = players_root / f"{player_id}.{suffix}"

            if self.output_format == "json":
                if player_file.exists():
                    existing = pd.read_json(player_file)

                    existing_event_ids = existing["id"].unique()
                    new_events = player_events[
                        ~player_events["id"].isin(existing_event_ids)
                    ]

                    if new_events.empty:
                        if logger.isEnabledFor(logging.DEBUG):
                            tqdm.write(
                                f"No new events for player {player_id} in match {match_id}; skipping write"
                            )
                        continue

                    combined = pd.concat([existing, new_events], ignore_index=True)
                    combined.to_json(player_file, index=False, orient="records")
                else:
                    player_events.to_json(player_file, index=False, orient="records")
            else:
                raise ValueError(
                    f"Unsupported output_format: {self.output_format}. Use json."
                )

        return player_file

    def write_normalized(
        self, match_id: int, events: pd.DataFrame, output_root: Path
    ) -> Path:

        suffix = "json" if self.output_format == "json" else self.output_format

        match_file = self.write_match(match_id, events, output_root, suffix)
        _ = self.write_players(match_id, events, output_root, suffix)

        return match_file

    def _resolve_output_path(self) -> Path:
        if self.output_path:
            return self.output_path
        if not self.data_path:
            raise FileNotFoundError("output_path or data_path is required")
        return self.data_path / "processed" / "events"

    def load_striker_ids(self, match_id: int) -> set[int]:
        if not self.data_path:
            if logger.isEnabledFor(logging.WARNING):
                tqdm.write("data_path missing; cannot resolve striker ids")
            return set()

        lineups_path = self.data_path / "raw" / "lineups" / f"{match_id}.json"
        if not lineups_path.exists():
            if logger.isEnabledFor(logging.WARNING):
                tqdm.write(f"Lineups file missing for match {match_id}")
            return set()

        lineups = pd.read_json(lineups_path)
        if "lineup" not in lineups.columns:
            if logger.isEnabledFor(logging.WARNING):
                tqdm.write(f"Lineups file missing lineup data for match {match_id}")
            return set()

        lineup_flat = lineups["lineup"].explode().dropna()
        lineup_norm = pd.json_normalize(lineup_flat)
        if "positions" not in lineup_norm.columns or "player_id" not in lineup_norm:
            if logger.isEnabledFor(logging.WARNING):
                tqdm.write(
                    f"Lineups file missing position details for match {match_id}"
                )
            return set()

        positions = lineup_norm[["player_id", "positions"]].explode("positions")
        positions = positions.dropna(subset=["positions"])
        if positions.empty:
            return set()

        positions_norm = pd.json_normalize(positions["positions"])
        positions_norm["player_id"] = positions["player_id"].values
        striker_rows = positions_norm[
            positions_norm["position_id"].isin(self.striker_position_ids)
        ]
        return set(striker_rows["player_id"].unique())
