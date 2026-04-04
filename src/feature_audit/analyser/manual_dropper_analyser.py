import pandas as pd

from .base import BaseAnalyzer, AnalyzerResult
from ..utils import drop_columns


class ManualDropAnalyzer(BaseAnalyzer):
    name = "manual_drop"

    def __init__(self, drop_map: dict[str, str]):
        self.drop_map = drop_map

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        columns = []
        for col, reason in self.drop_map.items():
            if col in df.columns:
                columns.append({"column": col, "reason": reason})
        columns.sort(key=lambda x: x["column"])
        return AnalyzerResult(name=self.name, payload={"columns": columns})

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
