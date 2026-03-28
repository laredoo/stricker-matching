"""Reference table ETL for StatsBomb data.

This is intentionally not wired into the training pipeline yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stricker_matching_model.etl.base import BaseETL


@dataclass
class RawStatsBombData:
    competitions: pd.DataFrame
    matches: pd.DataFrame
    lineups: pd.DataFrame


@dataclass
class ReferenceTables:
    players: pd.DataFrame
    positions: pd.DataFrame
    teams: pd.DataFrame


@dataclass
class StatsBombReferenceTablesETL(BaseETL[RawStatsBombData, ReferenceTables, None]):
    data_path: Path
    output_path: Path | None = None

    def extract(self) -> RawStatsBombData:
        return RawStatsBombData(
            competitions=self._extract_competitions(),
            matches=self._extract_matches(),
            lineups=self._extract_lineups(),
        )

    def transform(self, raw: RawStatsBombData) -> ReferenceTables:
        lineup_df = raw.lineups

        players_cols = {
            "player_id": "player_id",
            "player_name": "player_name",
            "player_nickname": "player_nickname",
            "jersey_number": "jersey_number",
            "country.id": "country_id",
            "country.name": "country",
            "team_id": "team_id",
            "team_name": "team_name",
        }

        players_df = lineup_df.rename(columns=players_cols)[
            list(players_cols.values())
        ].drop_duplicates()

        positions_flat = (
            lineup_df[["player_id", "positions"]]
            .explode("positions")
            .dropna(subset=["positions"])
        )
        positions_norm = pd.json_normalize(positions_flat["positions"])
        positions_norm["player_id"] = positions_flat["player_id"].values

        positions_first = positions_norm.dropna(
            subset=["position_id", "position"]
        ).drop_duplicates("player_id")
        players_df = players_df.merge(
            positions_first[["player_id", "position_id", "position"]],
            on="player_id",
            how="left",
        )
        players_df = players_df.drop_duplicates("player_id").reset_index(drop=True)

        positions_exploded = lineup_df["positions"].explode().dropna()
        positions_df = (
            pd.json_normalize(positions_exploded)
            .drop_duplicates(subset=["position_id", "position"])
            .reset_index(drop=True)
        )
        positions_df = positions_df[["position_id", "position"]]

        teams_df = players_df.drop_duplicates("team_name")[["team_id", "team_name"]]
        teams_df = teams_df.reset_index(drop=True)

        players_df = players_df[
            [
                "player_id",
                "player_name",
                "player_nickname",
                "jersey_number",
                "country_id",
                "country",
                "team_id",
                "position_id",
            ]
        ]

        return ReferenceTables(
            players=players_df, positions=positions_df, teams=teams_df
        )

    def load(self, transformed: ReferenceTables) -> None:
        output_path = self.output_path or self.data_path
        output_path.mkdir(parents=True, exist_ok=True)

        transformed.players.to_json(output_path / "players.json", orient="records")
        transformed.positions.to_json(output_path / "positions.json", orient="records")
        transformed.teams.to_json(output_path / "teams.json", orient="records")

    def run(self) -> ReferenceTables:
        raw = self.extract()
        tables = self.transform(raw)
        self.load(tables)
        return tables

    def _extract_competitions(self) -> pd.DataFrame:
        return pd.read_json(self.data_path / "competitions.json")

    def _extract_matches(self) -> pd.DataFrame:
        return pd.read_json(self.data_path / "matches.json")

    def _extract_lineups(self) -> pd.DataFrame:
        lineups_path = self.data_path / "lineups"
        lineup_files = sorted(lineups_path.glob("*.json"))
        lineup_frames: list[pd.DataFrame] = []

        for file_path in lineup_files:
            raw = pd.read_json(file_path)
            if isinstance(raw, pd.DataFrame) and "lineup" in raw.columns:
                raw_exploded = (
                    raw[["team_id", "team_name", "lineup"]]
                    .explode("lineup")
                    .dropna(subset=["lineup"])
                )
                lineup_norm = pd.json_normalize(raw_exploded["lineup"])
                lineup_norm["team_id"] = raw_exploded["team_id"].values
                lineup_norm["team_name"] = raw_exploded["team_name"].values
                lineup_frames.append(lineup_norm)

        if not lineup_frames:
            raise FileNotFoundError("No lineup files were processed")

        return pd.concat(lineup_frames, ignore_index=True)
