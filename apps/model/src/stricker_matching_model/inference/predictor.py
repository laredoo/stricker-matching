"""Inference utilities for the model service."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stricker_matching_model.core.artifacts import ArtifactStore


@dataclass
class Predictor:
    artifacts: ArtifactStore

    def predict(self, rows: list[list[float]]) -> list[int]:
        pipeline = self.artifacts.load()
        labels = pipeline.predict(np.array(rows))
        return [int(label) for label in labels]
