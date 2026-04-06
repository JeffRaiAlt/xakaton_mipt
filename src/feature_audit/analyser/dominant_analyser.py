import pandas as pd

from .base import BaseAnalyzer, AnalyzerResult
from ..utils import drop_columns, get_non_empty_series


class DominantValueAnalyzer(BaseAnalyzer):
    name = "dominant_value"

    def __init__(
        self,
        dominant_share_threshold: float = 1.0,
        min_non_empty_values: int = 3,
    ):
        self.dominant_share_threshold = dominant_share_threshold
        self.min_non_empty_values = min_non_empty_values

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        columns = []
        for col in df.columns:
            s = get_non_empty_series(df[col])
            non_empty_count = len(s)
            if non_empty_count < self.min_non_empty_values:
                continue

            value_counts = s.value_counts(dropna=False)
            top_value = value_counts.index[0]
            top_count = int(value_counts.iloc[0])
            top_share = float(top_count / non_empty_count)
            nunique = int(s.nunique(dropna=True))

            if top_share >= self.dominant_share_threshold:
                columns.append(
                    {
                        "column": col,
                        "top_value": top_value,
                        "top_count": top_count,
                        "non_empty_count": non_empty_count,
                        "top_share": round(top_share, 4),
                        "nunique": nunique,
                    }
                )

        columns.sort(key=lambda x: (-x["top_share"], x["nunique"], x["column"]))
        return AnalyzerResult(
            name=self.name,
            payload={
                "dominant_share_threshold": self.dominant_share_threshold,
                "min_non_empty_values": self.min_non_empty_values,
                "columns": columns,
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
