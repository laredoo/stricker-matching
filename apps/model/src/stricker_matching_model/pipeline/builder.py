"""Model pipeline builder."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stricker_matching_model.core.strategies import ClusteringStrategy


class PipelineBuilder:
    def build(self, strategy: ClusteringStrategy) -> Pipeline:
        # may include additional preprocessors here.
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("cluster", strategy.build_estimator()),
            ]
        )
