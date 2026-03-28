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
        self, players_list: list[int], players_df: dict[str, pd.DataFrame]
    ) -> list[pd.DataFrame]:
        return [
            self._calc_zone_event_proportions(players_list, players_df),
        ]

    def _calc_zone_event_proportions(
        self,
        players_list: list[int],
        players_df: dict[str, pd.DataFrame],
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

        all_events = pd.concat(players_df.values(), names=["player_id"])
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

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        return float(
            np.divide(numerator, denominator, out=np.array(0.0), where=denominator != 0)
        )
