from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.utils.feature_spec import FeatureSpec
from src.utils.io import load_feature_names_from_txt, load_specs_from_json


class FeatureProcessor:
    """
    Общий процессор для выделенной группы признаков.

    Публичный API:
        transform()

    В исследовательской фазе участник меняет private-методы,
    а потом переносит стабильную логику в production-версию.
    """

    def __init__(
        self,
        data_path: str | Path,
        feature_names_path: str | Path,
        group_name: str,
        owner: str,
        specs_json_path: Optional[str | Path] = None,
        sep: str = ",",
    ) -> None:
        self.data_path = Path(data_path)
        self.feature_names_path = Path(feature_names_path)
        self.group_name = group_name
        self.owner = owner
        self.specs_json_path = Path(specs_json_path) if specs_json_path else None
        self.sep = sep

    def transform(self) -> Tuple[pd.DataFrame, List[FeatureSpec]]:
        df = self._load_dataframe()
        feature_names = self._load_feature_names()
        self._validate_columns(df, feature_names)

        block_df = self._select_columns(df, feature_names)
        processed_df = self._process_features(block_df)

        external_specs = self._load_external_specs()
        specs = self._build_feature_specs(
            raw_block_df=block_df,
            processed_df=processed_df,
            external_specs=external_specs,
        )
        return processed_df, specs

    def _load_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path, sep=self.sep)

    def _load_feature_names(self) -> List[str]:
        return load_feature_names_from_txt(self.feature_names_path)

    def _load_external_specs(self) -> Dict[str, dict]:
        if self.specs_json_path is None:
            return {}
        loaded = load_specs_from_json(self.specs_json_path)
        mapping: Dict[str, dict] = {}
        for item in loaded:
            if isinstance(item, dict) and item.get("name"):
                mapping[str(item["name"])] = item
        return mapping

    def _validate_columns(self, df: pd.DataFrame, feature_names: List[str]) -> None:
        missing = [feature for feature in feature_names if feature not in df.columns]
        if missing:
            raise ValueError(
                f"Не найдены колонки для группы '{self.group_name}': {missing}"
            )

    def _select_columns(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        return df.loc[:, feature_names].copy()

    def _process_features(self, block_df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=block_df.index)

        for column in block_df.columns:
            if column == "lead_Ширина":
                self._add_width_feature(block_df, result)
            elif column == "lead_Линейная высота (см)":
                self._add_linear_height_feature(block_df, result)
            elif column == "lead_Вид оплаты":
                self._add_payment_type_feature(block_df, result)
            elif column == "returned_ts":
                self._add_returned_ts_feature(block_df, result)
            elif column == "lead_Служба доставки":
                self._add_delivery_service_feature(block_df, result)
            else:
                self._add_default_feature(block_df, result, column)

        return result

    def _add_width_feature(self, block_df: pd.DataFrame, result: pd.DataFrame) -> None:
        if "lead_Ширина" not in block_df.columns:
            return
        series = pd.to_numeric(block_df["lead_Ширина"], errors="coerce")
        result["lead_Ширина"] = series.fillna(series.median())

    def _add_linear_height_feature(self, block_df: pd.DataFrame, result: pd.DataFrame) -> None:
        if "lead_Линейная высота (см)" not in block_df.columns:
            return
        series = pd.to_numeric(block_df["lead_Линейная высота (см)"], errors="coerce")
        result["lead_Линейная высота (см)"] = series.fillna(series.median())
        result["lead_Линейная высота (см)__was_missing"] = series.isna().astype(int)

    def _add_payment_type_feature(self, block_df: pd.DataFrame, result: pd.DataFrame) -> None:
        if "lead_Вид оплаты" not in block_df.columns:
            return
        series = (
            block_df["lead_Вид оплаты"]
            .astype("string")
            .fillna("UNKNOWN")
            .str.strip()
            .str.lower()
        )
        result["lead_Вид оплаты"] = series.replace({"": "unknown"}).astype(str)

    def _add_returned_ts_feature(self, block_df: pd.DataFrame, result: pd.DataFrame) -> None:
        if "returned_ts" not in block_df.columns:
            return
        ts = pd.to_datetime(block_df["returned_ts"], errors="coerce")
        result["returned_ts"] = ts.astype("string")
        result["returned_ts__is_present"] = ts.notna().astype(int)

    def _add_delivery_service_feature(self, block_df: pd.DataFrame, result: pd.DataFrame) -> None:
        if "lead_Служба доставки" not in block_df.columns:
            return
        series = (
            block_df["lead_Служба доставки"]
            .astype("string")
            .fillna("UNKNOWN")
            .str.strip()
        )
        value_counts = series.value_counts(dropna=False)
        rare_values = value_counts[value_counts < 10].index
        result["lead_Служба доставки"] = series.replace(list(rare_values), "OTHER").astype(str)

    def _add_default_feature(self, block_df: pd.DataFrame, result: pd.DataFrame, column: str) -> None:
        series = block_df[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            result[column] = series.fillna("").astype(str)
        else:
            result[column] = series

    def _build_feature_specs(
        self,
        raw_block_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        external_specs: Dict[str, dict],
    ) -> List[FeatureSpec]:
        specs: List[FeatureSpec] = []

        for column in processed_df.columns:
            if column in external_specs:
                item = external_specs[column]
                specs.append(
                    FeatureSpec(
                        name=item.get("name", column),
                        source=item.get("source", column),
                        group=item.get("group", self.group_name),
                        description=item.get("description", f"Признак из блока {self.group_name}: {column}"),
                        baseline=bool(item.get("baseline", True)),
                        leakage_risk=item.get("leakage_risk", "none"),
                    )
                )
            else:
                specs.append(
                    FeatureSpec(
                        name=column,
                        source=column,
                        group=self.group_name,
                        description=f"Автоматически добавленный признак {column}",
                        baseline=True,
                        leakage_risk="none",
                    )
                )
        return specs
