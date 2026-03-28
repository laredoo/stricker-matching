"""Clustering strategy implementations.

Strategy pattern: the training pipeline depends on this interface, not on
specific algorithms. This lets you swap KMeans, HDBSCAN, etc. easily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans


class ClusteringStrategy(Protocol):
    def build_estimator(self) -> ClusterMixin: ...


@dataclass
class KMeansStrategy:  # ToDo: strategy is not the best patttern for multiple algorithms (e.g. KMeans, HDBSCAN, etc.) but good enough for first release
    n_clusters: int = 3
    random_state: int = 42

    def build_estimator(self) -> ClusterMixin:
        return KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
