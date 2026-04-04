import pandas as pd

from project2.src.feature_audit.analyser.base import BaseAnalyzer, AnalyzerResult
from project2.src.feature_audit.utils import calc_column_similarity, drop_columns, to_string_series


class DuplicateFeatureAnalyzer(BaseAnalyzer):
    name = "duplicate_features"

    def __init__(self, near_threshold: float = 0.95):
        self.near_threshold = near_threshold

    @staticmethod
    def are_columns_exact_duplicates(series_1: pd.Series, series_2: pd.Series) -> bool:
        s1 = to_string_series(series_1)
        s2 = to_string_series(series_2)
        return bool(s1.equals(s2))

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        exact_matches = []
        near_matches = []
        cols = list(df.columns)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_1 = cols[i]
                col_2 = cols[j]

                if self.are_columns_exact_duplicates(df[col_1], df[col_2]):
                    exact_matches.append(
                        {
                            "left": col_1,
                            "right": col_2,
                            "similarity": 1.0,
                        }
                    )
                    continue

                similarity = calc_column_similarity(df[col_1], df[col_2])
                if similarity >= self.near_threshold:
                    near_matches.append(
                        {
                            "left": col_1,
                            "right": col_2,
                            "similarity": round(similarity, 4),
                        }
                    )

        near_matches.sort(key=lambda x: (-x["similarity"], x["left"], x["right"]))
        return AnalyzerResult(
            name=self.name,
            payload={
                "near_threshold": self.near_threshold,
                "exact_matches": exact_matches,
                "near_matches": near_matches,
            },
        )

    def apply(
        self,
        df: pd.DataFrame,
        result: AnalyzerResult,
    ) -> tuple[pd.DataFrame, dict]:
        columns_to_drop = self.extract_exact_duplicate_columns_to_drop(result)
        work_df = drop_columns(df, columns_to_drop)
        return work_df, {"dropped_columns": columns_to_drop}

    @staticmethod
    def extract_exact_duplicate_columns_to_drop(result: AnalyzerResult) -> list[str]:
        cols = []
        for item in result.payload["exact_matches"]:
            if item["right"] not in cols:
                cols.append(item["right"])
        return cols
