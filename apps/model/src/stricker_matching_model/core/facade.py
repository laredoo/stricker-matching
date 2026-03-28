"""Facade for the model package.

Facade pattern: expose a stable train/predict interface that hides internal
modules (etl, features, pipeline, artifacts).
"""

from __future__ import annotations

from stricker_matching_model.inference.predictor import Predictor
from stricker_matching_model.train.trainer import BaseTrainer


class ModelFacade:
    def __init__(self, trainer: BaseTrainer, predictor: Predictor) -> None:
        self.trainer = trainer
        self.predictor = predictor

    def train(self) -> None:
        self.trainer.run()

    def predict(self, rows: list[list[float]]) -> list[int]:
        return self.predictor.predict(rows)
