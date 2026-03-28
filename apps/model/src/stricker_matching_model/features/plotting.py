from __future__ import annotations

from pathlib import Path
from sklearn.decomposition import PCA

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

    def plot_region_shift(
        self,
        events: pd.DataFrame,
        centroids: pd.DataFrame,
        viz_dir: Path | None,
        feature_name: str,
    ) -> None:
        if viz_dir is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib is not installed; skipping plots for %s",
                feature_name,
            )
            return

        viz_dir.mkdir(parents=True, exist_ok=True)

        for row in centroids.itertuples(index=False):
            player_id = row.player_id
            player_dir = viz_dir / self._player_id_dirname(player_id)
            player_dir.mkdir(parents=True, exist_ok=True)

            player_events = events[events["player_id"] == player_id]
            first_half = player_events[player_events["period"] == 1]
            final_30 = player_events[
                (player_events["period"] == 2) & (player_events["minute"] >= 60)
            ]

            fig, ax = plt.subplots(figsize=(5, 4))
            if not first_half.empty:
                ax.scatter(first_half["x"], first_half["y"], s=8, alpha=0.6, label="H1")
            if not final_30.empty:
                ax.scatter(
                    final_30["x"], final_30["y"], s=8, alpha=0.6, label="Final 30"
                )

            if pd.notna(row.x_c1) and pd.notna(row.y_c1):
                ax.scatter([row.x_c1], [row.y_c1], s=60, marker="x", label="c1")
            if pd.notna(row.x_c2) and pd.notna(row.y_c2):
                ax.scatter([row.x_c2], [row.y_c2], s=60, marker="x", label="c2")
            if (
                pd.notna(row.x_c1)
                and pd.notna(row.y_c1)
                and pd.notna(row.x_c2)
                and pd.notna(row.y_c2)
            ):
                ax.annotate(
                    "",
                    xy=(row.x_c2, row.y_c2),
                    xytext=(row.x_c1, row.y_c1),
                    arrowprops={"arrowstyle": "->", "color": "black"},
                )

            ax.set_title(f"{feature_name}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(player_dir / f"{feature_name}.png", dpi=150)
            plt.close(fig)

    def plot_territorial_shrinkage(
        self,
        events: pd.DataFrame,
        hulls: dict[int, np.ndarray],
        viz_dir: Path | None,
        feature_name: str,
    ) -> None:
        if viz_dir is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib is not installed; skipping plots for %s",
                feature_name,
            )
            return

        viz_dir.mkdir(parents=True, exist_ok=True)

        for player_id, hull in hulls.items():
            player_dir = viz_dir / self._player_id_dirname(player_id)
            player_dir.mkdir(parents=True, exist_ok=True)

            player_events = events[events["player_id"] == player_id]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(player_events["x"], player_events["y"], s=8, alpha=0.6)

            if hull.shape[0] >= 3:
                closed = np.vstack([hull, hull[0]])
                ax.plot(closed[:, 0], closed[:, 1], color="#e45756", linewidth=1.5)
                ax.fill(closed[:, 0], closed[:, 1], color="#e45756", alpha=0.15)
            elif hull.shape[0] == 2:
                ax.plot(hull[:, 0], hull[:, 1], color="#e45756", linewidth=1.5)
            elif hull.shape[0] == 1:
                ax.scatter(hull[0, 0], hull[0, 1], color="#e45756", s=30)

            ax.set_title(feature_name)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.tight_layout()
            fig.savefig(player_dir / f"{feature_name}.png", dpi=150)
            plt.close(fig)

    def plot_involvement_slope(
        self,
        events: pd.DataFrame,
        viz_dir: Path | None,
        n_bins: int,
        feature_name: str,
        involved_type_ids: set[int],
    ) -> None:
        if viz_dir is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib is not installed; skipping plots for %s",
                feature_name,
            )
            return

        filtered = events.copy()
        filtered["type_id"] = filtered["type"].map(
            lambda value: value.get("id") if isinstance(value, dict) else value
        )
        filtered = filtered[filtered["type_id"].isin(involved_type_ids)]
        if filtered.empty:
            return

        match_minute = filtered["minute"].astype(float)
        match_minute = match_minute + np.where(filtered["period"] == 2, 45.0, 0.0)
        filtered["match_minute"] = match_minute

        group_keys = ["player_id", "match_id"]
        start_minute = filtered.groupby(group_keys)["match_minute"].transform("min")
        end_minute = filtered.groupby(group_keys)["match_minute"].transform("max")
        duration = end_minute - start_minute
        bin_size = duration / float(n_bins)
        relative_minute = filtered["match_minute"] - start_minute
        relative_minute = relative_minute.clip(lower=0.0, upper=duration - 1e-6)
        filtered["bin_idx"] = (relative_minute / bin_size).astype(int)

        counts = (
            filtered.groupby(["player_id", "match_id", "bin_idx"])
            .size()
            .rename("count")
        )
        counts = counts.reset_index()
        counts = counts.pivot_table(
            index=["player_id", "match_id"],
            columns="bin_idx",
            values="count",
            fill_value=0,
        )
        counts = counts.reindex(columns=range(n_bins), fill_value=0)

        x_idx = np.arange(n_bins, dtype=float)
        x_fraction = (x_idx + 0.5) / float(n_bins)

        viz_dir.mkdir(parents=True, exist_ok=True)

        for player_id, player_counts in counts.groupby(level=0):
            y = player_counts.to_numpy(dtype=float)
            if y.size == 0:
                continue
            y_mean = y.mean(axis=0)
            beta1 = self._linear_regression_slope(x_idx, y_mean)
            beta0 = float(y_mean.mean()) - beta1 * float(x_idx.mean())
            y_fit = beta0 + beta1 * x_idx

            player_dir = viz_dir / self._player_id_dirname(player_id)
            player_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.scatter(x_fraction, y_mean, s=20, alpha=0.8)
            ax.plot(x_fraction, y_fit, color="#e45756", linewidth=1.5)
            ax.set_title(feature_name)
            ax.set_xlabel("playtime fraction")
            ax.set_ylabel("events per bin")
            fig.tight_layout()
            fig.savefig(player_dir / f"{feature_name}.png", dpi=150)
            plt.close(fig)

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

    def _linear_regression_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        denom = float(((x - x_mean) ** 2).sum())
        if denom == 0:
            return 0.0
        num = float(((x - x_mean) * (y - y_mean)).sum())
        return num / denom

    def plot_pca_variance(
        self, pca: PCA, plot_path: Path | None, viz_dir: Path | None
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not installed; skipping PCA plot")
            return

        if plot_path is None:
            if viz_dir is None:
                viz_dir = Path("data/features")
            plot_path = viz_dir / "pca" / "pca_variance.png"

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        ratios = pca.explained_variance_ratio_
        x = list(range(1, len(ratios) + 1))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x, ratios, color="#4C78A8", label="Explained variance")
        ax.plot(x, ratios.cumsum(), color="#F58518", marker="o", label="Cumulative")
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance ratio")
        ax.set_title("PCA explained variance")
        ax.set_xticks(x)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def plot_pca_cv(self, scores: dict[int, float], viz_dir: Path | None) -> None:
        if not scores:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not installed; skipping PCA CV plot")
            return

        if viz_dir is None:
            viz_dir = Path("data/features/viz")
        viz_dir.mkdir(parents=True, exist_ok=True)
        plot_path = viz_dir / "pca" / "pca_cv_error.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        components = sorted(scores.keys())
        errors = [scores[n] for n in components]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(components, errors, marker="o", color="#4C78A8")
        ax.set_xlabel("PCA components")
        ax.set_ylabel("CV reconstruction error")
        ax.set_title("PCA component selection (CV)")
        ax.set_xticks(components)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
