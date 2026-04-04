import pandas as pd

from project2.src.feature_audit.analyser.base import BaseAnalyzer, AnalyzerResult
from project2.src.feature_audit.utils import cramers_v, safe_binary_target, to_string_series


class CategoricalTargetCorrelationAnalyzer(BaseAnalyzer):
    name = "categorical_target_correlation"

    def __init__(
        self,
        target_column: str = "buyout_flag",
        min_non_null_count: int = 30,
        min_cramers_v: float = 0.05,
        max_unique_values: int = 100,
    ):
        self.target_column = target_column
        self.min_non_null_count = min_non_null_count
        self.min_cramers_v = min_cramers_v
        self.max_unique_values = max_unique_values

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        if self.target_column not in df.columns:
            return AnalyzerResult(
                name=self.name,
                payload={
                    "target_column": self.target_column,
                    "columns": [],
                    "error": "target_not_found",
                },
            )

        target = safe_binary_target(df[self.target_column]).astype("string")
        columns = []

        for col in df.columns:
            if col == self.target_column:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            values = to_string_series(df[col]).replace("", pd.NA)
            pair_mask = values.notna() & target.notna()
            pair_count = int(pair_mask.sum())
            if pair_count < self.min_non_null_count:
                continue

            nunique = int(values[pair_mask].nunique(dropna=True))
            if nunique <= 1 or nunique > self.max_unique_values:
                continue

            corr = cramers_v(values[pair_mask], target[pair_mask])
            if corr < self.min_cramers_v:
                continue

            columns.append(
                {
                    "column": col,
                    "pair_count": pair_count,
                    "nunique": nunique,
                    "cramers_v": round(float(corr), 4),
                }
            )

        columns.sort(key=lambda x: (-x["cramers_v"], x["column"]))
        return AnalyzerResult(
            name=self.name,
            payload={
                "target_column": self.target_column,
                "min_non_null_count": self.min_non_null_count,
                "min_cramers_v": self.min_cramers_v,
                "max_unique_values": self.max_unique_values,
                "columns": columns,
            },
        )
