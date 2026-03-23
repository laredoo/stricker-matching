"""Training workflow using Template Method pattern."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

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

    def run(self) -> None:
        raw = self.etl.extract()
        cleaned = self.etl.transform(raw)
        canonical = self.etl.load(cleaned)
        feature_rows = self.features.build(canonical)

        pipeline = self.pipeline_builder.build(self.strategy)
        pipeline.fit(np.array(feature_rows))
        self.artifacts.save(pipeline)
