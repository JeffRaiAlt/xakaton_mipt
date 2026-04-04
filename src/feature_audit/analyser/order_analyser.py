import pandas as pd

from project2.src.feature_audit.analyser.base import BaseAnalyzer, AnalyzerResult
from project2.src.feature_audit.utils import try_parse_datetime


class DateOrderAnalyzer(BaseAnalyzer):
    name = "date_order"

    def __init__(
        self,
        candidate_columns: list[str],
        pair_success_threshold: float = 0.80,
        min_pair_count: int = 30,
    ):
        self.candidate_columns = candidate_columns
        self.pair_success_threshold = pair_success_threshold
        self.min_pair_count = min_pair_count

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        parsed_columns = {}
        pair_stats = []
        score = {col: 0 for col in self.candidate_columns if col in df.columns}

        for col in self.candidate_columns:
            if col not in df.columns:
                continue
            parsed_columns[col] = self._try_parse_series(df[col])

        valid_columns = list(parsed_columns.keys())

        for i, col_a in enumerate(valid_columns):
            for j, col_b in enumerate(valid_columns):
                if i == j:
                    continue

                s_a = parsed_columns[col_a]
                s_b = parsed_columns[col_b]
                mask = s_a.notna() & s_b.notna()
                pair_count = int(mask.sum())
                if pair_count < self.min_pair_count:
                    continue

                a_vals = s_a[mask]
                b_vals = s_b[mask]
                le_share = float((a_vals <= b_vals).mean())
                lt_share = float((a_vals < b_vals).mean())
                eq_share = float((a_vals == b_vals).mean())

                pair_stats.append(
                    {
                        "left": col_a,
                        "right": col_b,
                        "pair_count": pair_count,
                        "le_share": round(le_share, 4),
                        "lt_share": round(lt_share, 4),
                        "eq_share": round(eq_share, 4),
                    }
                )

                if le_share >= self.pair_success_threshold:
                    score[col_a] -= 1
                    score[col_b] += 1

        ordered_columns = [
            item[0]
            for item in sorted(score.items(), key=lambda x: (x[1], x[0]))
        ]

        return AnalyzerResult(
            name=self.name,
            payload={
                "candidate_columns": valid_columns,
                "pair_success_threshold": self.pair_success_threshold,
                "min_pair_count": self.min_pair_count,
                "pair_stats": pair_stats,
                "column_scores": score,
                "ordered_columns": ordered_columns,
            },
        )

    @staticmethod
    def _try_parse_series(series: pd.Series) -> pd.Series:
        candidates = [
            try_parse_datetime(series),
            try_parse_datetime(series, unit="ms"),
            try_parse_datetime(series, unit="s"),
        ]
        return max(candidates, key=lambda x: x.notna().mean())
