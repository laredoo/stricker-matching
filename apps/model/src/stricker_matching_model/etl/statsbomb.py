"""StatsBomb ETL focused on producing model-ready rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from stricker_matching_model.etl.base import BaseETL


@dataclass
class StatsBombETL(BaseETL):
    """ETL for model training data.

    This class is intentionally scoped to preparing rows for the model pipeline.
    Reference-table generation lives in etl/reference_tables.py.
    """

    demo: bool = False

    def extract(self):
        if self.demo:
            return [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        raise NotImplementedError("Provide real training rows for demo=False")

    def transform(self, raw):
        return raw

    def load(self, transformed):
        return transformed
