from pathlib import Path

import numpy as np
import pandas as pd

from stricker_matching_model.logging import get_logger

logger = get_logger(__name__)


class FeatureBuilderContext:
    def __init__(
        self,
    ) -> None:
        self.logger = logger

        self._zone_ids = [1, 2, 3, 4, 5]
        self._event_type_ids = {
            42: "Ball Receipt",
            2: "Ball Recovery",
            3: "Dispossessed",
            4: "Duel",
            5: "Camera On*",
            6: "Block",
            8: "Offside",
            9: "Clearance",
            10: "Interception",
            14: "Dribble",
            16: "Shot",
            17: "Pressure",
            18: "Half Start*",
            19: "Substitution",
            20: "Own Goal Against",
            21: "Foul Won",
            22: "Foul Committed",
            23: "Goal Keeper",
            24: "Bad Behaviour",
            25: "Own Goal For",
            26: "Player On",
            27: "Player Off",
            28: "Shield",
            30: "Pass",
            33: "50/50",
            34: "Half End*",
            35: "Starting XI",
            36: "Tactical Shift",
            37: "Error",
            38: "Miscontrol",
            39: "Dribbled Past",
            40: "Injury Stoppage",
            41: "Referee Ball-Drop",
            43: "Carry",
        }

    def calculate_features(
        self, players_list: list[int], all_events: pd.DataFrame
    ) -> list[pd.DataFrame]:
        return [
            self._calc_zone_event_proportions(players_list, all_events),
            self._calc_pass_outcome_proportions(players_list, all_events),
            self._calc_shot_features(players_list, all_events),
        ]

    def _calc_zone_event_proportions(
        self,
        players_list: list[int],
        all_events: pd.DataFrame,
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

        return features.reset_index()

    def _calc_pass_outcome_proportions(
        self,
        players_list: list[int],
        all_events: pd.DataFrame,
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
        return proportions.reset_index()

    def _calc_shot_features(
        self,
        players_list: list[int],
        all_events: pd.DataFrame,
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

        return features.reset_index()

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
        not_open_play = shot_events[~shot_events["shot_type_name"] == "Open Play"]
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
