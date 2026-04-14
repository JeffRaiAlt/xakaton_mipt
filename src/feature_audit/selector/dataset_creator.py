from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from feature_audit.selector.selector import FeatureSelector
from feature_audit.selector.base import FeatureSelectionStrategy
from feature_audit.selector.strategies.weighted_voting import WeightedVotingSelectionStrategy
from feature_audit.selector.strategies.weighted_rank_conflict import WeightedRankConflictStrategy


@dataclass
class DatasetReducer:
    data_path: str | Path
    target: str = "buyout_flag"
    time_col: str = "lead_created_dt"
    start_date: str = "2025-03-01"
    end_date: str = "2026-03-29"
    always_keep: tuple[str, ...] = ("sale_ts", "buyout_flag")
    drop_if_exists: tuple[str, ...] = ("lead_group_quality",)

    def reduce(self, selected_features: Sequence[str]) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)

        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
        try:
            df[self.time_col] = df[self.time_col].dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass

        df = df[df[self.time_col].between(self.start_date, self.end_date)].copy()

        final_cols = list(dict.fromkeys(list(selected_features) + list(self.always_keep)))

        missing_cols = [col for col in final_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in dataset: {missing_cols}")

        df = df[final_cols].copy()

        cols_to_drop = [col for col in self.drop_if_exists if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df

def do_work(
    strategy: FeatureSelectionStrategy,
    data_path: str | Path,
    output_path: str | Path,
) -> None:

    selector = FeatureSelector(strategy)
    df_selected = selector.select()

    print(df_selected.head(23))
    print(df_selected.shape)

    selected_feature_names = df_selected["feature"].tolist()

    reducer = DatasetReducer(data_path=data_path)
    df_final = reducer.reduce(selected_feature_names)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    print("Final dataset shape:", df_final.shape)