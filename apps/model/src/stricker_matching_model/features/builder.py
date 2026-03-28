"""Feature engineering utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tqdm import tqdm

import pandas as pd

from stricker_matching_model.base.feature_builder_context import FeatureBuilderContext


class FeatureBuilder(FeatureBuilderContext):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._players_list = []

        self._players_path = None
        self._matches_path = None

        self._players_files_path_list = []
        self._matches_files_path_list = []

        self._players = dict[str, pd.DataFrame()]
        self._matches = dict[str, pd.DataFrame()]

        self._features = pd.DataFrame()
        self._all_players_events = pd.DataFrame()
        self._all_matches_events = pd.DataFrame()

    def build(
        self,
        data: Iterable[Path],
        output_path: Path = Path("features.json"),
        plot_features: bool = False,
        plot_pca: bool = False,
        pca_components: int = 10,
        viz_dir: Path | None = None,
    ) -> list[list[float]]:
        self.logger.info("Starting to build features...")

        if viz_dir is None:
            viz_dir = Path("data/features/viz")

        self._populate_variables(data)

        features = self._calculate_features(
            plot_features=plot_features,
            plot_pca=plot_pca,
            pca_components=pca_components,
            viz_dir=viz_dir,
        )

        self._save_features(features, output_path)

        return features.drop(columns=["player_id"]).to_numpy().tolist()

    def _calculate_features(
        self,
        plot_features: bool = False,
        plot_pca: bool = False,
        pca_components: int = 10,
        viz_dir: Path | None = None,
    ) -> pd.DataFrame:

        feature_blocks = self.calculate_features(
            self._players_list,
            self._all_players_events,
            plot_features=plot_features,
            viz_dir=viz_dir,
        )

        features = self._features.set_index("player_id")

        for block in feature_blocks:
            features = features.join(block.set_index("player_id"), how="left")

        features = self._apply_pca(
            features.reset_index(),
            pca_components=pca_components,
            plot_pca=plot_pca,
            viz_dir=viz_dir,
        )

        return features

    def _player_id_from_file(self, file_path: Path) -> int:
        try:
            return int(file_path.stem)
        except ValueError:
            raise ValueError(f"Player file name is not an int id: {file_path.name}")

    def _get_ids(self, path: Path) -> list[int]:
        return [int(file.stem) for file in sorted(path.glob("*.json"))]

    def _get_data_path(self, path: Path) -> tuple[Path, Path, list[Path], list[Path]]:
        if not path.exists():
            raise ValueError(f"Data path {path} not found")

        players_path = path / "processed" / "events" / "players"
        _players_files_path_list = list(sorted(players_path.glob("*.json")))

        if not players_path.exists():
            raise ValueError(f"Players path {players_path} not found")

        matches_path = path / "processed" / "events" / "matches"
        _matches_files_path_list = list(sorted(matches_path.glob("*.json")))

        if not matches_path.exists():
            raise ValueError(f"Matches path {matches_path} not found")

        return (
            players_path,
            matches_path,
            _players_files_path_list,
            _matches_files_path_list,
        )

    def _populate_variables(self, data: Iterable[Path]) -> None:
        (
            self._players_path,
            self._matches_path,
            self._players_files_path_list,
            self._matches_files_path_list,
        ) = self._get_data_path(data)

        self._players_list = self._get_ids(self._players_path)
        self._matches_list = self._get_ids(self._matches_path)

        self._features = pd.DataFrame({"player_id": self._players_list})

        self.logger.info("Reading player data")
        self._players = {
            str(player_id): pd.read_json(file_path)
            for player_id, file_path in tqdm(
                zip(self._players_list, self._players_files_path_list),
                total=len(self._players_list),
            )
        }
        self._all_players_events = pd.concat(
            [self._players[str(player_id)] for player_id in self._players_list],
            names=["player_id"],
        )

        self.logger.info("Reading matches data")
        self._matches = {
            str(match_id): pd.read_json(file_path)
            for match_id, file_path in tqdm(
                zip(self._matches_list, self._matches_files_path_list),
                total=len(self._matches_files_path_list),
            )
        }
        self._all_matches_events = pd.concat(self._matches.values(), names=["match_id"])

    def _save_features(self, features: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_json(output_path, orient="records", lines=True)
