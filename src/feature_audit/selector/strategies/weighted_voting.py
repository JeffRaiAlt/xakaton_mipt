
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from ..base import FeatureSelectionStrategy


@dataclass
class WeightedVotingSelectionStrategy(FeatureSelectionStrategy):
    logreg_path: str | Path
    rf_path: str | Path
    catboost_path: str | Path

    top_k_logreg: int = 60
    top_k_rf: int = 60
    top_k_cb: int = 60

    final_top_n: int = 40
    min_votes: int = 2

    w_logreg: float = 40.0
    w_rf: float = 10.0
    w_cb: float = 50.0

    def select(self) -> pd.DataFrame:
        df_lr_raw = pd.read_csv(self.logreg_path)
        df_rf_raw = pd.read_csv(self.rf_path)
        df_cb_raw = pd.read_csv(self.catboost_path)

        df_lr = self._prepare_rank_scores(
            df=df_lr_raw,
            feature_col="base_feature",
            score_col="abs_coef",
            top_k=self.top_k_logreg,
            prefix="lr",
        )

        df_rf = self._prepare_rank_scores(
            df=df_rf_raw,
            feature_col="feature",
            score_col="importance",
            top_k=self.top_k_rf,
            prefix="rf",
        )

        df_cb = self._prepare_rank_scores(
            df=df_cb_raw,
            feature_col="feature",
            score_col="final_importance",
            top_k=self.top_k_cb,
            prefix="cb",
        )

        df_final = df_lr.merge(df_rf, on="feature", how="outer")
        df_final = df_final.merge(df_cb, on="feature", how="outer")

        score_cols = [
            "rank_lr", "score_lr", "vote_lr",
            "rank_rf", "score_rf", "vote_rf",
            "rank_cb", "score_cb", "vote_cb",
        ]
        for col in score_cols:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(0)

        df_final["votes"] = df_final["vote_lr"] + df_final["vote_rf"] + df_final["vote_cb"]
        df_final["final_score"] = (
            self.w_logreg * df_final["score_lr"]
            + self.w_rf * df_final["score_rf"]
            + self.w_cb * df_final["score_cb"]
        )

        df_final = df_final.sort_values(
            by=["votes", "final_score"],
            ascending=[False, False],
        ).reset_index(drop=True)

        df_selected = df_final[df_final["votes"] >= self.min_votes].copy()

        if len(df_selected) < self.final_top_n:
            selected_set = set(df_selected["feature"])
            df_extra = df_final[~df_final["feature"].isin(selected_set)].copy()
            n_to_add = self.final_top_n - len(df_selected)
            df_selected = pd.concat([df_selected, df_extra.head(n_to_add)], ignore_index=True)

        df_selected = df_selected.sort_values(
            by=["votes", "final_score"],
            ascending=[False, False],
        ).head(self.final_top_n).reset_index(drop=True)

        return df_selected

    @staticmethod
    def _prepare_rank_scores(
        df: pd.DataFrame,
        feature_col: str,
        score_col: str,
        top_k: int,
        prefix: str,
    ) -> pd.DataFrame:
        temp = df[[feature_col, score_col]].copy()
        temp = temp.dropna(subset=[feature_col])
        temp[feature_col] = temp[feature_col].astype(str)

        temp = temp.sort_values(score_col, ascending=False).drop_duplicates(
            subset=[feature_col], keep="first"
        ).head(top_k).reset_index(drop=True)

        temp[f"rank_{prefix}"] = np.arange(1, len(temp) + 1)
        temp[f"score_{prefix}"] = (top_k - temp[f"rank_{prefix}"] + 1) / top_k
        temp[f"vote_{prefix}"] = 1

        temp = temp.rename(columns={feature_col: "feature"})
        return temp[["feature", f"rank_{prefix}", f"score_{prefix}", f"vote_{prefix}"]]