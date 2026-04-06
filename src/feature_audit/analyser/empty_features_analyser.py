import pandas as pd

from .base import BaseAnalyzer, AnalyzerResult
from ..utils import calc_empty_share, drop_columns


class EmptyFeatureAnalyzer(BaseAnalyzer):
    name = "empty_features"

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        empty_cols = []
        for col in df.columns:
            empty_share = calc_empty_share(df[col])
            if empty_share >= self.threshold:
                empty_cols.append(
                    {
                        "column": col,
                        "empty_share": round(empty_share, 4),
                    }
                )
        return AnalyzerResult(
            name=self.name,
            payload={
                "threshold": self.threshold,
                "columns": empty_cols,
            },
        )

    def apply(
        self,
        df: pd.DataFrame,
        result: AnalyzerResult,
    ) -> tuple[pd.DataFrame, dict]:
        columns_to_drop = self.extract_column_names(result)
        work_df = drop_columns(df, columns_to_drop)
        return work_df, {"dropped_columns": columns_to_drop}

    @staticmethod
    def extract_column_names(result: AnalyzerResult) -> list[str]:
        return [item["column"] for item in result.payload["columns"]]
