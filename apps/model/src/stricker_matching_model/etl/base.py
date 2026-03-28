"""Base ETL interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseETL(ABC):
    """Abstract ETL contract.

    Extract: read raw sources.
    Transform: build canonical outputs.
    Load: persist outputs to a target store.
    """

    @abstractmethod
    def extract(self):
        raise NotImplementedError

    @abstractmethod
    def transform(self, raw):
        raise NotImplementedError

    @abstractmethod
    def load(self, transformed):
        raise NotImplementedError
