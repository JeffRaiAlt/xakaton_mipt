import pandas as pd

from project2.src.feature_audit.analyser.base import BaseAnalyzer, AnalyzerResult
from project2.src.feature_audit.utils import safe_binary_target, safe_numeric


class NumericTargetCorrelationAnalyzer(BaseAnalyzer):
    name = "numeric_target_correlation"

    def __init__(
        self,
        target_column: str = "buyout_flag",
        min_non_null_count: int = 30,
        min_abs_correlation: float = 0.05,
    ):
        self.target_column = target_column
        self.min_non_null_count = min_non_null_count
        self.min_abs_correlation = min_abs_correlation

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

        target = safe_binary_target(df[self.target_column])
        columns = []

        for col in df.columns:
            if col == self.target_column:
                continue
            numeric = safe_numeric(df[col])
            mask = numeric.notna() & target.notna()
            pair_count = int(mask.sum())
            if pair_count < self.min_non_null_count:
                continue
            corr = numeric[mask].corr(target[mask], method="pearson")
            if pd.isna(corr):
                continue
            if abs(float(corr)) < self.min_abs_correlation:
                continue
            columns.append(
                {
                    "column": col,
                    "pair_count": pair_count,
                    "correlation": round(float(corr), 4),
                    "abs_correlation": round(abs(float(corr)), 4),
                }
            )

        columns.sort(key=lambda x: (-x["abs_correlation"], x["column"]))
        return AnalyzerResult(
            name=self.name,
            payload={
                "target_column": self.target_column,
                "min_non_null_count": self.min_non_null_count,
                "min_abs_correlation": self.min_abs_correlation,
                "columns": columns,
            },
        )
