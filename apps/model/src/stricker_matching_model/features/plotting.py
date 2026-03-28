from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stricker_matching_model.logging import get_logger

logger = get_logger(__name__)


class FeaturePlotter:
    def plot_zone_event_proportions(
        self,
        features: pd.DataFrame,
        viz_dir: Path | None,
    ) -> None:
        self._plot_feature_distributions(
            features,
            viz_dir,
            title_prefix="zone_event_proportions",
            bins=20,
        )

    def plot_pass_outcome_proportions(
        self,
        features: pd.DataFrame,
        viz_dir: Path | None,
    ) -> None:
        self._plot_feature_distributions(
            features,
            viz_dir,
            title_prefix="pass_outcome_proportions",
            bins=20,
        )

    def plot_shot_features(
        self,
        features: pd.DataFrame,
        viz_dir: Path | None,
    ) -> None:
        self._plot_feature_distributions(
            features,
            viz_dir,
            title_prefix="shot_features",
            bins=20,
        )

    def _plot_feature_distributions(
        self,
        features: pd.DataFrame,
        viz_dir: Path | None,
        title_prefix: str,
        bins: int,
    ) -> None:
        if viz_dir is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib is not installed; skipping plots for %s",
                title_prefix,
            )
            return

        plot_columns = [col for col in features.columns if col != "player_id"]
        if not plot_columns:
            return

        viz_dir.mkdir(parents=True, exist_ok=True)

        for col in plot_columns:
            series = features[col].dropna()
            for _, row in features.iterrows():
                player_id = row["player_id"]
                player_dir = viz_dir / self._player_id_dirname(player_id)
                player_dir.mkdir(parents=True, exist_ok=True)

                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.hist(series, bins=bins, alpha=0.7, color="#4c78a8")
                ax.axvline(row[col], color="#e45756", linestyle="--", linewidth=1.5)
                ax.set_title(f"{title_prefix} - {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("count")
                fig.tight_layout()
                fig.savefig(player_dir / f"{col}.png", dpi=150)
                plt.close(fig)

    def _player_id_dirname(self, player_id: object) -> str:
        if pd.isna(player_id):
            return "unknown"
        try:
            return str(int(player_id))
        except (TypeError, ValueError):
            return str(player_id)
