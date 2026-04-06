import pandas as pd

from .base import BaseAnalyzer, AnalyzerResult
from ..utils import safe_numeric


class NumericFeatureCorrelationAnalyzer(BaseAnalyzer):
    name = "numeric_feature_correlation"

    def __init__(
        self,
        min_abs_correlation: float = 0.80,
        min_pair_count: int = 30,
    ):
        self.min_abs_correlation = min_abs_correlation
        self.min_pair_count = min_pair_count

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        numeric_cols = []
        numeric_df = pd.DataFrame(index=df.index)

        for col in df.columns:
            numeric_series = safe_numeric(df[col])
            if int(numeric_series.notna().sum()) >= self.min_pair_count:
                numeric_df[col] = numeric_series
                numeric_cols.append(col)

        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                left = numeric_cols[i]
                right = numeric_cols[j]
                mask = numeric_df[left].notna() & numeric_df[right].notna()
                pair_count = int(mask.sum())
                if pair_count < self.min_pair_count:
                    continue

                corr = numeric_df.loc[mask, left].corr(numeric_df.loc[mask, right])
                if pd.isna(corr):
                    continue
                abs_corr = abs(float(corr))
                if abs_corr < self.min_abs_correlation:
                    continue

                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "pair_count": pair_count,
                        "correlation": round(float(corr), 4),
                        "abs_correlation": round(abs_corr, 4),
                    }
                )

        pairs.sort(key=lambda x: (-x["abs_correlation"], x["left"], x["right"]))
        return AnalyzerResult(
            name=self.name,
            payload={
                "min_abs_correlation": self.min_abs_correlation,
                "min_pair_count": self.min_pair_count,
                "pairs": pairs,
            },
        )
