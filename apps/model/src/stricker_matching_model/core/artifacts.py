"""Artifact persistence.

Joblib is used here to serialize the trained sklearn pipeline. The model service
loads it at startup or on first request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def default_model_artifact_path() -> Path:
    return _repo_root() / "data" / "artifacts" / "model.joblib"


def default_model_output_path() -> Path:
    return _repo_root() / "data" / "model_output.json"


class ArtifactStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, pipeline: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.path)

    def load(self) -> Any:
        if not self.path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.path}")
        return joblib.load(self.path)
