import pandas as pd
from .base import BaseAnalyzer, AnalyzerResult


from ..utils import drop_columns, get_non_empty_series


class HighCardinalityAnalyzer(BaseAnalyzer):
    name = "high_cardinality"

    def __init__(
        self,
        unique_share_threshold: float = 0.90,
        min_unique_values: int = 50,
        only_string_like: bool = True,
    ):
        self.unique_share_threshold = unique_share_threshold
        self.min_unique_values = min_unique_values
        self.only_string_like = only_string_like

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        columns = []
        for col in df.columns:
            if self.only_string_like:
                if not (
                    pd.api.types.is_object_dtype(df[col])
                    or pd.api.types.is_string_dtype(df[col])
                    or isinstance(df[col].dtype, pd.CategoricalDtype)
                ):
                    continue

            s = get_non_empty_series(df[col])
            non_empty_count = len(s)
            if non_empty_count == 0:
                continue

            nunique = int(s.nunique(dropna=True))
            unique_share = float(nunique / non_empty_count)

            if (
                nunique >= self.min_unique_values
                and unique_share >= self.unique_share_threshold
            ):
                columns.append(
                    {
                        "column": col,
                        "nunique": nunique,
                        "non_empty_count": non_empty_count,
                        "unique_share": round(unique_share, 4),
                    }
                )

        columns.sort(key=lambda x: (-x["unique_share"], -x["nunique"], x["column"]))
        return AnalyzerResult(
            name=self.name,
            payload={
                "unique_share_threshold": self.unique_share_threshold,
                "min_unique_values": self.min_unique_values,
                "only_string_like": self.only_string_like,
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
