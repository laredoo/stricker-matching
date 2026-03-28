from pathlib import Path

import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from stricker_matching_model.logging import get_logger
from stricker_matching_model.features.plotting import FeaturePlotter

logger = get_logger(__name__)


class FeatureBuilderContext:
    def __init__(
        self,
    ) -> None:
        self.logger = logger
        self._plotter = FeaturePlotter()

        self._zone_ids = [1, 2, 3, 4, 5]
        self._pca_columns = [
            "proport_ball_receipt_zone_1",
            "proport_dribble_zone_1",
            "proport_pressure_zone_1",
            "proport_shot_zone_1",
            "proport_pass_zone_1",
            "proport_dribbled_past_zone_1",
            "proport_interception_zone_1",
            "proport_ball_receipt_zone_2",
            "proport_dribble_zone_2",
            "proport_pressure_zone_2",
            "proport_shot_zone_2",
            "proport_pass_zone_2",
            "proport_dribbled_past_zone_2",
            "proport_interception_zone_2",
            "proport_ball_receipt_zone_3",
            "proport_dribble_zone_3",
            "proport_pressure_zone_3",
            "proport_shot_zone_3",
            "proport_pass_zone_3",
            "proport_dribbled_past_zone_3",
            "proport_interception_zone_3",
            "proport_ball_receipt_zone_4",
            "proport_dribble_zone_4",
            "proport_pressure_zone_4",
            "proport_shot_zone_4",
            "proport_pass_zone_4",
            "proport_dribbled_past_zone_4",
            "proport_interception_zone_4",
            "proport_ball_receipt_zone_5",
            "proport_dribble_zone_5",
            "proport_pressure_zone_5",
            "proport_shot_zone_5",
            "proport_pass_zone_5",
            "proport_dribbled_past_zone_5",
            "proport_interception_zone_5",
            "proport_pass_short_success",
            "proport_pass_long_success",
            "proport_pass_unsuccessful",
            "proport_shot_all_events",
            "proport_shot_goal",
            "proport_shot_on_target",
            "proport_not_long_shot",
            "proport_head_all_shots",
            "proport_shot_not_open_play",
        ]

    def calculate_features(
        self,
        players_list: list[int],
        all_events: pd.DataFrame,
        plot_features: bool = False,
        viz_dir: Path | None = None,
    ) -> list[pd.DataFrame]:
        return [
            self._calc_territorial_shrinkage(
                players_list, all_events, plot_features, viz_dir
            ),
            self._calc_involvement_slope(
                players_list, all_events, plot_features, viz_dir
            ),
            self._calc_region_shift_delta_x(
                players_list, all_events, plot_features, viz_dir
            ),
            self._calc_zone_event_proportions(
                players_list, all_events, plot_features, viz_dir
            ),
            self._calc_pass_outcome_proportions(
                players_list, all_events, plot_features, viz_dir
            ),
            self._calc_shot_features(players_list, all_events, plot_features, viz_dir),
        ]

    def _calc_zone_event_proportions(
        self,
        players_list: list[int],
        all_events: pd.DataFrame,
        plot_features: bool = False,
        viz_dir: Path | None = None,
    ) -> pd.DataFrame:

        interested_event_types = {
            42: "ball_receipt",
            14: "dribble",
            17: "pressure",
            16: "shot",
            30: "pass",
            39: "dribbled_past",
            10: "interception",
        }

        total_events = all_events.groupby("player_id").size()

        events_by_zone = all_events.dropna(subset=["pitch_zone"]).copy()
        events_by_zone["type_id"] = events_by_zone["type"].str.get("id")
        events_by_zone = events_by_zone[
            events_by_zone["type_id"].isin(interested_event_types)
        ]

        counts = events_by_zone.groupby(["player_id", "pitch_zone", "type_id"]).size()
        full_index = pd.MultiIndex.from_product(
            [players_list, self._zone_ids, list(interested_event_types.keys())],
            names=["player_id", "pitch_zone", "type_id"],
        )
        counts = counts.reindex(full_index, fill_value=0)

        proportions = counts.div(total_events.reindex(players_list), level="player_id")
        proportions = proportions.fillna(0)

        features = proportions.unstack(["pitch_zone", "type_id"])
        features.columns = [
            f"proport_{interested_event_types[type_id]}_zone_{zone}"
            for zone, type_id in features.columns
        ]

        features = features.reset_index()

        if plot_features:
            self._plotter.plot_zone_event_proportions(features, viz_dir)

        return features

    def _calc_pass_outcome_proportions(
        self,
        players_list: list[int],
        all_events: pd.DataFrame,
        plot_features: bool = False,
        viz_dir: Path | None = None,
    ) -> pd.DataFrame:
        short_pass_max_length = 15.0

        pass_events = all_events[all_events["type"].str.get("id") == 30].copy()
        pass_events["pass_length"] = pass_events["pass"].str.get("length")
        pass_events["pass_outcome"] = pass_events["pass"].str.get("outcome")

        total_passes = pass_events.groupby("player_id").size()
        total_passes = total_passes.reindex(players_list, fill_value=0)

        success_mask = pass_events["pass_outcome"].isna()
        short_mask = pass_events["pass_length"] <= short_pass_max_length
        long_mask = pass_events["pass_length"] > short_pass_max_length

        short_success = (
            pass_events[success_mask & short_mask].groupby("player_id").size()
        )
        long_success = pass_events[success_mask & long_mask].groupby("player_id").size()
        unsuccessful = pass_events[~success_mask].groupby("player_id").size()

        counts = pd.DataFrame(
            {
                "proport_pass_short_success": short_success,
                "proport_pass_long_success": long_success,
                "proport_pass_unsuccessful": unsuccessful,
            }
        ).reindex(players_list, fill_value=0)

        proportions = counts.div(total_passes.replace(0, np.nan), axis=0).fillna(0)
        proportions.index.name = "player_id"
        features = proportions.reset_index()

        if plot_features:
            self._plotter.plot_pass_outcome_proportions(features, viz_dir)

        return features

    def _calc_shot_features(
        self,
        players_list: list[int],
        all_events: pd.DataFrame,
        plot_features: bool = False,
        viz_dir: Path | None = None,
    ) -> pd.DataFrame:
        shot_events = self._prepare_shot_events(all_events)

        total_events = all_events.groupby("player_id").size()
        total_events = total_events.reindex(players_list, fill_value=0)

        total_shots = shot_events.groupby("player_id").size()
        total_shots = total_shots.reindex(players_list, fill_value=0)

        features = pd.DataFrame(index=players_list)
        features.index.name = "player_id"
        features["proport_shot_all_events"] = (
            total_shots / total_events.replace(0, np.nan)
        ).fillna(0)
        features["proport_shot_goal"] = self._calc_shot_goal(
            shot_events, players_list, total_shots
        )
        features["proport_shot_on_target"] = self._calc_shot_on_target(
            shot_events, players_list, total_shots
        )
        features["proport_not_long_shot"] = self._calc_not_long_shot(
            shot_events, players_list, total_shots
        )
        features["proport_head_all_shots"] = self._calc_head_shot(
            shot_events, players_list, total_shots
        )
        features["proport_shot_not_open_play"] = self._calc_shot_not_open_play(
            shot_events, players_list, total_shots
        )
        features["avg_statsbomb_xg"] = self._calc_shot_xg(shot_events, players_list)

        features = features.reset_index()

        if plot_features:
            self._plotter.plot_shot_features(features, viz_dir)

        return features

    def _calc_region_shift_delta_x(
        self,
        players_list: list[int],
        all_events: pd.DataFrame,
        plot_features: bool = False,
        viz_dir: Path | None = None,
    ) -> pd.DataFrame:
        events = all_events.dropna(subset=["x", "y", "minute"]).copy()

        first_half = events[events["period"] == 1]
        final_30 = events[(events["period"] == 2) & (events["minute"] >= 60)]

        c1 = first_half.groupby("player_id")[["x", "y"]].mean()
        c2 = final_30.groupby("player_id")[["x", "y"]].mean()

        centroids = c1.join(c2, how="outer", lsuffix="_c1", rsuffix="_c2")
        centroids = centroids.reindex(players_list)
        centroids["delta_x_region_shift"] = centroids["x_c2"] - centroids["x_c1"]

        features = centroids[["delta_x_region_shift"]].fillna(0)
        features = features.reset_index()

        if plot_features:
            self._plotter.plot_region_shift(
                events,
                centroids.reset_index(),
                viz_dir,
                feature_name="delta_x_region_shift",
            )

        return features

    def _calc_involvement_slope(
        self,
        players_list: list[int],
        all_events: pd.DataFrame,
        plot_features: bool = False,
        viz_dir: Path | None = None,
    ) -> pd.DataFrame:
        n_bins = 6
        involved_type_ids = [
            42,
            2,
            3,
            4,
            6,
            8,
            9,
            10,
            14,
            16,
            17,
            21,
            22,
            24,
            25,
            28,
            30,
            33,
            34,
            38,
            39,
            43,
        ]
        events = all_events.dropna(
            subset=["minute", "period", "type", "match_id"]
        ).copy()
        events["type_id"] = events["type"].map(
            lambda value: value.get("id") if isinstance(value, dict) else value
        )
        events = events[events["type_id"].isin(involved_type_ids)]

        if events.empty:
            features = pd.DataFrame({"player_id": players_list})
            features["involvement_slope_beta1"] = 0.0
            return features

        match_minute = events["minute"].astype(float)
        match_minute = match_minute + np.where(events["period"] == 2, 45.0, 0.0)
        events["match_minute"] = match_minute

        group_keys = ["player_id", "match_id"]
        start_minute = events.groupby(group_keys)["match_minute"].transform("min")
        end_minute = events.groupby(group_keys)["match_minute"].transform("max")
        duration = end_minute - start_minute
        bin_size = duration / float(n_bins)
        relative_minute = events["match_minute"] - start_minute
        relative_minute = relative_minute.clip(lower=0.0, upper=duration - 1e-6)
        events["bin_idx"] = (relative_minute / bin_size).astype(int)

        counts = (
            events.groupby(["player_id", "match_id", "bin_idx"]).size().rename("count")
        )
        counts = counts.reset_index()
        counts = counts.pivot_table(
            index=["player_id", "match_id"],
            columns="bin_idx",
            values="count",
            fill_value=0,
        )
        counts = counts.reindex(columns=range(n_bins), fill_value=0)

        x = np.arange(n_bins, dtype=float)
        x_mean = float(x.mean())
        denom = float(((x - x_mean) ** 2).sum())
        if denom == 0:
            slopes = pd.Series(0.0, index=counts.index)
        else:
            y = counts.to_numpy(dtype=float)
            y_mean = y.mean(axis=1, keepdims=True)
            num = ((x - x_mean) * (y - y_mean)).sum(axis=1)
            slopes = pd.Series(num / denom, index=counts.index)

        slopes = (
            slopes.groupby(level="player_id").mean().reindex(players_list, fill_value=0)
        )

        features = pd.DataFrame({"player_id": players_list})
        features["involvement_slope_beta1"] = slopes.values

        if plot_features:
            self._plotter.plot_involvement_slope(
                events,
                viz_dir,
                n_bins=n_bins,
                feature_name="involvement_slope_beta1",
                involved_type_ids=involved_type_ids,
            )

        return features

    def _calc_territorial_shrinkage(
        self,
        players_list: list[int],
        all_events: pd.DataFrame,
        plot_features: bool = False,
        viz_dir: Path | None = None,
    ) -> pd.DataFrame:
        events = all_events.dropna(subset=["x", "y"]).copy()
        keep_quantile = 0.75

        areas: dict[int, float] = {}
        hulls: dict[int, np.ndarray] = {}

        for player_id, group in events.groupby("player_id"):
            points = group[["x", "y"]].to_numpy(dtype=float)
            points = self._filter_spatial_outliers(points, keep_quantile)
            hull = self._convex_hull(points)
            hulls[int(player_id)] = hull
            areas[int(player_id)] = self._polygon_area(hull)

        features = pd.DataFrame({"player_id": players_list})
        features["territorial_shrinkage_area"] = (
            pd.Series(areas).reindex(players_list).fillna(0).values
        )

        if plot_features:
            self._plotter.plot_territorial_shrinkage(
                events,
                hulls,
                viz_dir,
                feature_name="territorial_shrinkage_area",
            )

        return features

    def _convex_hull(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        unique = np.unique(points, axis=0)
        if len(unique) <= 2:
            return unique

        sorted_points = sorted(unique.tolist())

        def cross(o: list[float], a: list[float], b: list[float]) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[list[float]] = []
        for p in sorted_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: list[list[float]] = []
        for p in reversed(sorted_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        return np.array(hull, dtype=float)

    def _polygon_area(self, polygon: np.ndarray) -> float:
        if polygon.shape[0] < 3:
            return 0.0
        x = polygon[:, 0]
        y = polygon[:, 1]
        return 0.5 * float(
            np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        )

    def _filter_spatial_outliers(
        self,
        points: np.ndarray,
        keep_quantile: float,
    ) -> np.ndarray:
        if points.size == 0:
            return points
        median = np.median(points, axis=0)
        distances = np.sqrt(((points - median) ** 2).sum(axis=1))
        cutoff = np.quantile(distances, keep_quantile)
        return points[distances <= cutoff]

    def _linear_regression_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        denom = float(((x - x_mean) ** 2).sum())
        if denom == 0:
            return 0.0
        num = float(((x - x_mean) * (y - y_mean)).sum())
        return num / denom

    def _prepare_shot_events(self, all_events: pd.DataFrame) -> pd.DataFrame:
        shot_events = all_events[all_events["type"].str.get("id") == 16].copy()
        shot_events["shot_outcome"] = shot_events["shot"].str.get("outcome")
        shot_events["shot_outcome_name"] = shot_events["shot_outcome"].str.get("name")
        shot_events["shot_type"] = shot_events["shot"].str.get("type")
        shot_events["shot_type_name"] = shot_events["shot_type"].str.get("name")
        shot_events["shot_body_part"] = shot_events["shot"].str.get("body_part")
        shot_events["shot_body_part_name"] = shot_events["shot_body_part"].str.get(
            "name"
        )
        shot_events["shot_xg"] = shot_events["shot"].str.get("statsbomb_xg")
        shot_events["play_pattern_name"] = shot_events["play_pattern"].str.get("name")

        # Shot distance to goal center (StatsBomb pitch uses 120x80 coordinates).
        shot_events["shot_distance"] = np.sqrt(
            (120.0 - shot_events["x"]) ** 2 + (40.0 - shot_events["y"]) ** 2
        )
        return shot_events

    def _calc_shot_goal(
        self,
        shot_events: pd.DataFrame,
        players_list: list[int],
        total_shots: pd.Series,
    ) -> pd.Series:
        goals = shot_events[shot_events["shot_outcome_name"] == "Goal"]
        counts = goals.groupby("player_id").size().reindex(players_list, fill_value=0)
        return (counts / total_shots.replace(0, np.nan)).fillna(0)

    def _calc_shot_on_target(
        self,
        shot_events: pd.DataFrame,
        players_list: list[int],
        total_shots: pd.Series,
    ) -> pd.Series:
        on_target_outcomes = {"Goal", "Saved", "Saved to Post", "Post"}
        on_target = shot_events[
            shot_events["shot_outcome_name"].isin(on_target_outcomes)
        ]
        counts = (
            on_target.groupby("player_id").size().reindex(players_list, fill_value=0)
        )
        return (counts / total_shots.replace(0, np.nan)).fillna(0)

    def _calc_not_long_shot(
        self,
        shot_events: pd.DataFrame,
        players_list: list[int],
        total_shots: pd.Series,
    ) -> pd.Series:
        long_shot_min_distance = 25.0
        not_long_shots = shot_events[
            shot_events["shot_distance"] < long_shot_min_distance
        ]
        counts = (
            not_long_shots.groupby("player_id")
            .size()
            .reindex(players_list, fill_value=0)
        )
        return (counts / total_shots.replace(0, np.nan)).fillna(0)

    def _calc_head_shot(
        self,
        shot_events: pd.DataFrame,
        players_list: list[int],
        total_shots: pd.Series,
    ) -> pd.Series:
        headers = shot_events[shot_events["shot_body_part_name"] == "Head"]
        counts = headers.groupby("player_id").size().reindex(players_list, fill_value=0)
        return (counts / total_shots.replace(0, np.nan)).fillna(0)

    def _calc_shot_not_open_play(
        self,
        shot_events: pd.DataFrame,
        players_list: list[int],
        total_shots: pd.Series,
    ) -> pd.Series:
        not_open_play = shot_events[shot_events["shot_type_name"] != "Open Play"]
        counts = (
            not_open_play.groupby("player_id")
            .size()
            .reindex(players_list, fill_value=0)
        )
        return (counts / total_shots.replace(0, np.nan)).fillna(0)

    def _calc_shot_xg(
        self,
        shot_events: pd.DataFrame,
        players_list: list[int],
    ) -> pd.Series:
        avg_xg = shot_events.groupby("player_id")["shot_xg"].mean()
        return avg_xg.reindex(players_list, fill_value=0)

    def _apply_pca(
        self,
        features: pd.DataFrame,
        pca_components: int,
        plot_pca: bool,
        viz_dir: Path,
        random_state: int = 42,
    ) -> pd.DataFrame:
        pca_plot_path = viz_dir / "pca" / "pca_variance.png"

        available = [col for col in self._pca_columns if col in features.columns]
        missing = [col for col in self._pca_columns if col not in features.columns]
        if missing:
            self.logger.warning("Missing PCA columns: %s", missing)
        if not available:
            return features

        x = features[available].fillna(0.0).to_numpy(dtype=float)
        max_components = min(max(2, pca_components), x.shape[1])
        best_components, cv_scores = self._select_pca_components(
            x,
            max_components=max_components,
            random_state=random_state,
        )

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        pca = PCA(n_components=best_components, random_state=random_state)
        pca_values = pca.fit_transform(x_scaled)
        pca_df = pd.DataFrame(
            pca_values,
            columns=[f"pca_{i + 1}" for i in range(pca_values.shape[1])],
        )

        features = pd.concat([features.drop(columns=available), pca_df], axis=1)

        pca_artifact_path = Path("../../data/artifacts/pca.joblib")
        pca_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": scaler,
                "pca": pca,
                "columns": available,
                "n_components": best_components,
            },
            pca_artifact_path,
        )

        if plot_pca:
            self._plotter.plot_pca_variance(pca, pca_plot_path, viz_dir)
            self._plotter.plot_pca_cv(cv_scores, viz_dir)

        return features

    def _pca_cv_errors(
        self,
        x: np.ndarray,
        max_components: int,
        random_state: int,
    ) -> dict[int, float]:
        n_samples = x.shape[0]
        n_splits = min(5, n_samples) if n_samples > 1 else 1
        if n_splits < 2:
            return {max_components: 0.0}

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores: dict[int, float] = {}

        for n_components in range(2, max_components + 1):
            fold_errors: list[float] = []
            for train_idx, test_idx in kfold.split(x):
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x[train_idx])
                x_test = scaler.transform(x[test_idx])

                pca = PCA(n_components=n_components, random_state=random_state)
                pca.fit(x_train)
                x_recon = pca.inverse_transform(pca.transform(x_test))
                fold_errors.append(float(np.mean((x_test - x_recon) ** 2)))
            scores[n_components] = float(np.mean(fold_errors))

        return scores

    def _pca_elbow_choice(self, scores: dict[int, float]) -> int:
        components = np.array(sorted(scores.keys()), dtype=float)
        errors = np.array([scores[int(c)] for c in components], dtype=float)

        if len(components) <= 2:
            return int(components[-1])

        p1 = np.array([components[0], errors[0]])
        p2 = np.array([components[-1], errors[-1]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return int(components[-1])

        line_unit = line_vec / line_len
        points = np.column_stack([components, errors])
        projections = p1 + np.outer((points - p1) @ line_unit, line_unit)
        distances = np.linalg.norm(points - projections, axis=1)
        elbow_idx = int(np.argmax(distances))
        return int(components[elbow_idx])

    def _select_pca_components(
        self, x: np.ndarray, max_components: int, random_state: int
    ) -> tuple[int, dict[int, float]]:
        scores = self._pca_cv_errors(
            x,
            max_components=max_components,
            random_state=random_state,
        )

        if not scores:
            return max_components, {max_components: 0.0}

        return self._pca_elbow_choice(scores), scores
