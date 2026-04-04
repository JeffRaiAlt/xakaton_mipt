import re
import warnings

import pandas as pd

from .base import BaseAnalyzer, AnalyzerResult


class DateCandidateAnalyzer(BaseAnalyzer):
    name = "date_candidates"

    DATE_NAME_HINTS = (
        "date", "dt", "time",
        "created", "updated", "closed",
        "returned", "received", "paid",
    )

    DATE_STRING_PATTERN = re.compile(
        r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"
    )

    def __init__(
        self,
        parse_success_threshold: float = 0.80,
        sample_size: int = 300,
        ts_suffix: str = "_ts",
    ):
        self.parse_success_threshold = parse_success_threshold
        self.sample_size = sample_size
        self.ts_suffix = ts_suffix

    def _safe_to_datetime(self, series: pd.Series, **kwargs) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(series, errors="coerce", **kwargs)

    def _has_date_name_hint(self, col: str) -> bool:
        col_lower = col.lower()
        if col_lower.endswith(self.ts_suffix):
            return True
        return any(hint in col_lower for hint in self.DATE_NAME_HINTS)

    def _analyze_numeric_timestamp(self, col: str, sample: pd.Series) -> dict | None:
        if not self._has_date_name_hint(col):
            return None

        numeric = pd.to_numeric(sample, errors="coerce")
        numeric_share = float(numeric.notna().mean())
        if numeric_share < self.parse_success_threshold:
            return None

        valid_s_mask = numeric.between(946684800, 2082758400)
        valid_ms_mask = numeric.between(946684800000, 2082758400000)

        s_share = float(valid_s_mask.mean())
        ms_share = float(valid_ms_mask.mean())

        best_unit = "ms" if ms_share >= s_share else "s"
        best_share = max(ms_share, s_share)
        if best_share < self.parse_success_threshold:
            return None

        return {
            "kind": "timestamp",
            "best_unit": best_unit,
            "parse_success_share": round(best_share, 4),
            "numeric_share": round(numeric_share, 4),
        }

    def _analyze_string_date(self, sample: pd.Series) -> dict | None:
        sample_str = sample.astype("string").str.strip()
        pattern_mask = sample_str.str.match(self.DATE_STRING_PATTERN, na=False)
        pattern_share = float(pattern_mask.mean())
        if pattern_share < 0.50:
            return None

        parsed = self._safe_to_datetime(sample_str)
        success_share = float(parsed.notna().mean())
        if success_share < self.parse_success_threshold:
            return None

        return {
            "kind": "string_date",
            "best_unit": None,
            "parse_success_share": round(success_share, 4),
            "pattern_share": round(pattern_share, 4),
        }

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        ts_suffix_columns = []
        detected_date_columns = []

        for col in df.columns:
            series = df[col]
            if col.endswith(self.ts_suffix):
                ts_suffix_columns.append(col)

            non_null = series.dropna()
            if len(non_null) == 0:
                continue

            sample = non_null.head(self.sample_size)
            numeric_result = self._analyze_numeric_timestamp(col, sample)
            string_result = self._analyze_string_date(sample)

            best_result = None
            if numeric_result and string_result:
                if numeric_result["parse_success_share"] >= string_result["parse_success_share"]:
                    best_result = numeric_result
                else:
                    best_result = string_result
            elif numeric_result:
                best_result = numeric_result
            elif string_result:
                best_result = string_result

            if best_result is None:
                continue

            detected_date_columns.append(
                {
                    "column": col,
                    "has_ts_suffix": col.endswith(self.ts_suffix),
                    **best_result,
                }
            )

        detected_date_columns.sort(key=lambda x: (-x["parse_success_share"], x["column"]))
        return AnalyzerResult(
            name=self.name,
            payload={
                "ts_suffix": self.ts_suffix,
                "parse_success_threshold": self.parse_success_threshold,
                "ts_suffix_columns": sorted(ts_suffix_columns),
                "detected_date_columns": detected_date_columns,
            },
        )
