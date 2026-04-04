from __future__ import annotations

import json
from pathlib import Path
from typing import Any


import numpy as np
import pandas as pd
#from sklearn.model_selection import KFold


EMPTY_TOKENS = frozenset({"", "nan", "none", "null", "nat"})


def normalize_column_name(name: str) -> str:
    """Нормализует имя колонки для сравнения и логирования."""
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def drop_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    if not columns_to_drop:
        return df.copy()
    return df.drop(columns=columns_to_drop, errors="ignore").copy()


def to_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def get_non_empty_series(series: pd.Series) -> pd.Series:
    s = to_string_series(series)
    mask = ~s.str.lower().isin(EMPTY_TOKENS)
    return s[mask]


def calc_empty_share(series: pd.Series) -> float:
    s = to_string_series(series)
    mask = s.str.lower().isin(EMPTY_TOKENS)
    return float(mask.mean())


def save_report(report: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2, default=str)


def calc_column_similarity(series_1: pd.Series, series_2: pd.Series) -> float:
    s1 = to_string_series(series_1)
    s2 = to_string_series(series_2)

    comparable_mask = ~((s1 == "") & (s2 == ""))
    comparable_count = int(comparable_mask.sum())
    if comparable_count == 0:
        return 0.0

    similarity = (s1[comparable_mask] == s2[comparable_mask]).mean()
    return float(similarity)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

"""def add_oof_target_encoding(
    df,
    col,
    target,
    n_splits=5,
    random_state=42,
    new_col=None
):
    
    df = df.copy()
    new_col = new_col or f"{col}_te"

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    df[new_col] = 0.0

    global_mean = df[target].mean()

    for tr_idx, val_idx in kf.split(df):
        means = df.iloc[tr_idx].groupby(col)[target].mean()
        df.loc[val_idx, new_col] = df.loc[val_idx, col].map(means)

    df[new_col].fillna(global_mean, inplace=True)

    return df"""


def safe_binary_target(series: pd.Series) -> pd.Series:
    """Пытается привести target к бинарному 0/1 виду."""
    numeric = pd.to_numeric(series, errors="coerce")
    unique_values = set(numeric.dropna().unique().tolist())
    if unique_values.issubset({0, 1}):
        return numeric.astype("float64")

    string = to_string_series(series).str.lower()
    mapping = {
        "0": 0.0,
        "1": 1.0,
        "false": 0.0,
        "true": 1.0,
        "no": 0.0,
        "yes": 1.0,
    }
    mapped = string.map(mapping)
    return mapped.astype("float64")


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V без SciPy, через chi2 по contingency table."""
    valid_mask = x.notna() & y.notna()
    if int(valid_mask.sum()) == 0:
        return 0.0

    table = pd.crosstab(x[valid_mask], y[valid_mask])
    if table.empty:
        return 0.0

    observed = table.to_numpy(dtype="float64")
    n = observed.sum()
    if n == 0:
        return 0.0

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)

    r, k = observed.shape
    denom = min(r - 1, k - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / denom))


def try_parse_datetime(series: pd.Series, unit: str | None = None) -> pd.Series:
    if unit is None:
        return pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(numeric, unit=unit, errors="coerce")


def build_step_report(
    *,
    analyzer_name: str,
    action: str,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    result: dict[str, Any],
    transform: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "analyzer": analyzer_name,
        "action": action,
        "before_shape": list(before_shape),
        "after_shape": list(after_shape),
        "result": result,
        "transform": transform or {},
    }
