import pandas as pd

from .base import BaseAnalyzer, AnalyzerResult


class DateNormalizationAnalyzer(BaseAnalyzer):
    name = "date_normalization"

    def __init__(self, candidate_specs: list[dict]):
        self.candidate_specs = candidate_specs

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        columns = []
        for item in self.candidate_specs:
            col = item["column"]
            if col not in df.columns:
                continue
            columns.append(
                {
                    "column": col,
                    "new_column": self._build_date_column_name(col),
                    "kind": item.get("kind"),
                    "best_unit": item.get("best_unit"),
                }
            )
        return AnalyzerResult(name=self.name, payload={"columns": columns})

    def apply(
        self,
        df: pd.DataFrame,
        result: AnalyzerResult,
    ) -> tuple[pd.DataFrame, dict]:
        work_df = df.copy()
        dropped_columns = []
        created_columns = []

        for item in result.payload["columns"]:
            old_col = item["column"]
            new_col = item["new_column"]
            best_unit = item.get("best_unit")
            kind = item.get("kind")

            if kind == "timestamp" and best_unit in {"s", "ms"}:
                numeric = pd.to_numeric(work_df[old_col], errors="coerce")
                work_df[new_col] = pd.to_datetime(numeric, unit=best_unit, errors="coerce")
            else:
                work_df[new_col] = pd.to_datetime(work_df[old_col], errors="coerce")

            created_columns.append(new_col)
            dropped_columns.append(old_col)

        work_df = work_df.drop(columns=dropped_columns, errors="ignore")
        return work_df, {
            "created_columns": created_columns,
            "dropped_columns": dropped_columns,
        }

    @staticmethod
    def _build_date_column_name(col: str) -> str:
        if col.endswith("_ts"):
            return col[:-3] + "_dt"
        if col.endswith("_at"):
            return col[:-3] + "_dt"
        if col.endswith("_date"):
            return col + "_dt"
        if col.endswith("_dt"):
            return col
        return col + "_dt"
