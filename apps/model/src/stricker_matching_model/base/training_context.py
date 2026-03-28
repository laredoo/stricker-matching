from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from stricker_matching_model.core.artifacts import default_cluster_plot_path
from stricker_matching_model.logging import get_logger

try:  # Matplotlib is optional in some environments.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - import guarded for optional dependency
    plt = None

logger = get_logger(__name__)


@dataclass
class TrainingContext:
    output_path: Path

    def save_outputs(self, labels: np.ndarray, player_ids: pd.Series | None) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if player_ids is None:
            output = pd.DataFrame({"label": labels.astype(int)})
        else:
            output = pd.DataFrame(
                {"player_id": player_ids.astype(int), "label": labels.astype(int)}
            )
        output.to_json(self.output_path, orient="records", lines=True)

    def log_metrics(self, pipeline: Any, rows: np.ndarray, labels: np.ndarray) -> None:
        cluster_step = pipeline.named_steps.get("cluster")
        scaler_step = pipeline.named_steps.get("scaler")
        scaled = scaler_step.transform(rows) if scaler_step is not None else rows

        inertia = getattr(cluster_step, "inertia_", None)
        if inertia is not None:
            logger.info("KMeans inertia: %.4f", float(inertia))
        else:
            logger.warning("Inertia not available on cluster estimator")

        unique_labels = np.unique(labels)
        n_clusters = unique_labels.size
        n_samples = scaled.shape[0]
        if n_clusters < 2 or n_clusters >= n_samples:
            logger.warning(
                "Invalid clustering for silhouette/CH metrics (clusters=%s, samples=%s)",
                n_clusters,
                n_samples,
            )
            return

        silhouette = silhouette_score(scaled, labels)
        calinski = calinski_harabasz_score(scaled, labels)
        logger.info("Silhouette coefficient: %.4f", float(silhouette))
        logger.info("Calinski-Harabasz index: %.4f", float(calinski))

    def plot_clusters(
        self, pipeline: Any, rows: np.ndarray, labels: np.ndarray
    ) -> None:
        if plt is None:
            logger.warning("matplotlib not installed; skipping PCA cluster plot")
            return

        scaler_step = pipeline.named_steps.get("scaler")
        scaled = scaler_step.transform(rows) if scaler_step is not None else rows
        coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
        plot_path = default_cluster_plot_path()
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=12)
        ax.set_title("Cluster projection (PCA 2D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(*scatter.legend_elements(), title="cluster", loc="best")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        logger.info("Saved cluster PCA plot to %s", plot_path)
