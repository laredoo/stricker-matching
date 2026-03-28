"""Training workflow using Template Method pattern."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from stricker_matching_model.core.artifacts import ArtifactStore
from stricker_matching_model.core.strategies import ClusteringStrategy
from stricker_matching_model.etl.statsbomb import StatsBombETL
from stricker_matching_model.features.builder import FeatureBuilder
from stricker_matching_model.pipeline.builder import PipelineBuilder


class BaseTrainer(Protocol):
    def run(self) -> None: ...


@dataclass
class StatsBombTrainer:  # ToDo: not the best architecture for multiple providers, but good enough for first release.
    etl: StatsBombETL
    features: FeatureBuilder
    pipeline_builder: PipelineBuilder
    strategy: ClusteringStrategy
    artifacts: ArtifactStore
    output_path: Path

    def run(self) -> None:
        raw = self.etl.extract()
        cleaned = self.etl.transform(raw)
        canonical = self.etl.load(cleaned)
        feature_rows = self.features.build(canonical)
        rows = np.array(feature_rows)
        pipeline = self.pipeline_builder.build(self.strategy)
        pipeline.fit(rows)
        self.artifacts.save(pipeline)
        labels = pipeline.predict(rows)
        self._save_outputs(labels, None)

    def _save_outputs(self, labels: np.ndarray, player_ids: pd.Series | None) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if player_ids is None:
            output = pd.DataFrame({"label": labels.astype(int)})
        else:
            output = pd.DataFrame(
                {"player_id": player_ids.astype(int), "label": labels.astype(int)}
            )
        output.to_json(self.output_path, orient="records", lines=True)


@dataclass
class FeatureTrainer:
    features_path: Path
    pipeline_builder: PipelineBuilder
    strategy: ClusteringStrategy
    artifacts: ArtifactStore
    output_path: Path

    def run(self) -> None:
        features = pd.read_json(self.features_path, lines=True)
        if features.empty:
            raise ValueError(f"No feature rows found in {self.features_path}")
        player_ids = features.get("player_id")
        rows = features.drop(columns=["player_id"], errors="ignore").to_numpy()
        pipeline = self.pipeline_builder.build(self.strategy)
        pipeline.fit(rows)
        self.artifacts.save(pipeline)
        labels = pipeline.predict(rows)
        self._save_outputs(labels, player_ids)

    def _save_outputs(self, labels: np.ndarray, player_ids: pd.Series | None) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if player_ids is None:
            output = pd.DataFrame({"label": labels.astype(int)})
        else:
            output = pd.DataFrame(
                {"player_id": player_ids.astype(int), "label": labels.astype(int)}
            )
        output.to_json(self.output_path, orient="records", lines=True)
