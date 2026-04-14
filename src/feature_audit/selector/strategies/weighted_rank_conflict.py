from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from feature_audit.selector.base import FeatureSelectionStrategy



@dataclass
class WeightedRankConflictStrategy(FeatureSelectionStrategy):
    logreg_path: str | Path
    rf_path: str | Path
    catboost_path: str | Path

    top_k: int = 30
    final_top_n: int = 40

    w_logreg: int = 40
    w_rf: int = 10
    w_cb: int = 50

    random_state: int = 42

    def select(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)

        lr_features = self._load_top_features(
            path=self.logreg_path,
            feature_col="base_feature",
            score_col="max_abs_coef",
        )
        rf_features = self._load_top_features(
            path=self.rf_path,
            feature_col="feature",
            score_col="importance",
        )
        cb_features = self._load_top_features(
            path=self.catboost_path,
            feature_col="feature",
            score_col="importance",
        )

        lr_features = self._shuffle_features(lr_features, rng)
        rf_features = self._shuffle_features(rf_features, rng)
        cb_features = self._shuffle_features(cb_features, rng)

        selected = []
        selected_set = set()

        max_len = max(len(lr_features), len(rf_features), len(cb_features))

        for rank_idx in range(max_len):
            candidates = []

            if rank_idx < len(lr_features):
                candidates.append(("lr", lr_features[rank_idx], self.w_logreg))
            if rank_idx < len(rf_features):
                candidates.append(("rf", rf_features[rank_idx], self.w_rf))
            if rank_idx < len(cb_features):
                candidates.append(("cb", cb_features[rank_idx], self.w_cb))

            if not candidates:
                continue

            # удаляем дубли по feature, оставляя вариант с наибольшим весом
            best_by_feature = {}
            for source, feature, weight in candidates:
                if feature not in best_by_feature or weight > best_by_feature[feature][1]:
                    best_by_feature[feature] = (source, weight)

            unique_candidates = [
                (source, feature, weight)
                for feature, (source, weight) in best_by_feature.items()
            ]

            # сортировка по весу убыв., чтобы первым шел самый приоритетный
            unique_candidates.sort(key=lambda x: x[2], reverse=True)

            for source, feature, weight in unique_candidates:
                if feature not in selected_set:
                    selected.append(
                        {
                            "feature": feature,
                            "source": source,
                            "source_weight": weight,
                            "rank_position": rank_idx + 1,
                        }
                    )
                    selected_set.add(feature)
                    break

            if len(selected) >= self.final_top_n:
                break

        # если не добрали — дополняем остатками по весовому приоритету
        if len(selected) < self.final_top_n:
            leftovers = self._build_leftovers(
                lr_features=lr_features,
                rf_features=rf_features,
                cb_features=cb_features,
                already_selected=selected_set,
            )

            for row in leftovers:
                if row["feature"] not in selected_set:
                    selected.append(row)
                    selected_set.add(row["feature"])

                if len(selected) >= self.final_top_n:
                    break

        return pd.DataFrame(selected)

    def _load_top_features(
        self,
        path: str | Path,
        feature_col: str,
        score_col: str,
    ) -> list[str]:
        df = pd.read_csv(path)

        temp = df[[feature_col, score_col]].copy()
        temp = temp.dropna(subset=[feature_col])
        temp[feature_col] = temp[feature_col].astype(str)

        temp = (
            temp.sort_values(score_col, ascending=False)
            .drop_duplicates(subset=[feature_col], keep="first")
            .head(self.top_k)
            .reset_index(drop=True)
        )

        return temp[feature_col].tolist()

    @staticmethod
    def _shuffle_features(features: list[str], rng: np.random.Generator) -> list[str]:
        shuffled = features.copy()
        rng.shuffle(shuffled)
        return shuffled

    def _build_leftovers(
        self,
        lr_features: list[str],
        rf_features: list[str],
        cb_features: list[str],
        already_selected: set[str],
    ) -> list[dict]:
        rows = []

        for i, feature in enumerate(cb_features, start=1):
            if feature not in already_selected:
                rows.append(
                    {
                        "feature": feature,
                        "source": "cb",
                        "source_weight": self.w_cb,
                        "rank_position": i,
                    }
                )

        for i, feature in enumerate(lr_features, start=1):
            if feature not in already_selected:
                rows.append(
                    {
                        "feature": feature,
                        "source": "lr",
                        "source_weight": self.w_logreg,
                        "rank_position": i,
                    }
                )

        for i, feature in enumerate(rf_features, start=1):
            if feature not in already_selected:
                rows.append(
                    {
                        "feature": feature,
                        "source": "rf",
                        "source_weight": self.w_rf,
                        "rank_position": i,
                    }
                )

        return rows