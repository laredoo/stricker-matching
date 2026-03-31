from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import stricker_matching_model.base.training_context as training_context
from stricker_matching_model.base.training_context import TrainingContext
from stricker_matching_model.core.artifacts import ArtifactStore
from stricker_matching_model.core.strategies import KMeansStrategy
from stricker_matching_model.pipeline.builder import PipelineBuilder
from stricker_matching_model.train.trainer import FeatureTrainer


def _write_features(path: Path) -> None:
    rows = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "f1": [0.0, 1.0, 0.1, 0.9],
            "f2": [1.0, 0.0, 0.9, 0.2],
        }
    )
    rows.to_json(path, orient="records", lines=True)


def test_training_context_logs_and_plots_without_matplotlib(tmp_path: Path) -> None:
    training_context.plt = None

    rows = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.9, 0.1]])
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("cluster", KMeans(n_clusters=2, random_state=0)),
        ]
    )
    pipeline.fit(rows)
    labels = pipeline.predict(rows)

    context = TrainingContext(output_path=tmp_path / "model_output.json")
    context.log_metrics(pipeline, rows, labels)
    context.plot_clusters(pipeline, rows, labels)


def test_feature_trainer_writes_artifact_and_output(tmp_path: Path) -> None:
    features_path = tmp_path / "features.json"
    _write_features(features_path)

    artifact_path = tmp_path / "model.joblib"
    output_path = tmp_path / "model_output.json"

    trainer = FeatureTrainer(
        output_path=output_path,
        features_path=features_path,
        pipeline_builder=PipelineBuilder(),
        strategy=KMeansStrategy(n_clusters=2, random_state=0),
        artifacts=ArtifactStore(artifact_path),
    )
    trainer.run()

    assert artifact_path.exists()
    assert output_path.exists()
