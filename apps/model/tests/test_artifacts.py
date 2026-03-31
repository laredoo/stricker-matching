from __future__ import annotations

from pathlib import Path

from stricker_matching_model.core.artifacts import (
    ArtifactStore,
    default_cluster_plot_path,
    default_model_artifact_path,
    default_model_output_path,
)


def test_artifact_store_save_load_roundtrip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "model.joblib")
    payload = {"k": 3, "name": "demo"}

    store.save(payload)
    loaded = store.load()

    assert loaded == payload


def test_default_artifact_paths_live_under_data_artifacts() -> None:
    model_path = default_model_artifact_path()
    output_path = default_model_output_path()
    plot_path = default_cluster_plot_path()

    assert str(model_path).endswith("data/artifacts/model.joblib")
    assert str(output_path).endswith("data/artifacts/model_output.json")
    assert str(plot_path).endswith("data/artifacts/cluster_pca_2d.png")
