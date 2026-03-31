from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stricker_matching_model.core.strategies import KMeansStrategy
from stricker_matching_model.pipeline.builder import PipelineBuilder


def test_kmeans_strategy_builds_estimator() -> None:
    strategy = KMeansStrategy()
    estimator = strategy.build_estimator()

    assert isinstance(estimator, KMeans)
    assert estimator.n_clusters == 3


def test_pipeline_builder_wires_scaler_and_cluster() -> None:
    builder = PipelineBuilder()
    pipeline = builder.build(KMeansStrategy(n_clusters=2))

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps.keys()) == ["scaler", "cluster"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["cluster"], KMeans)
