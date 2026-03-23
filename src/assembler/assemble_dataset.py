from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.feature_spec import FeatureSpec


def assemble_dataset(
    processors,
    output_dataset_path: str | Path | None = None,
    output_feature_spec_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Запускает несколько FeatureProcessor и собирает итоговый датасет.
    """
    blocks: List[pd.DataFrame] = []
    all_specs: List[FeatureSpec] = []

    for processor in processors:
        X_block, specs = processor.transform()
        blocks.append(X_block)
        all_specs.extend(specs)

    final_dataset = pd.concat(blocks, axis=1) if blocks else pd.DataFrame()
    feature_spec_df = pd.DataFrame([spec.to_dict() for spec in all_specs])

    if not feature_spec_df.empty and feature_spec_df["name"].duplicated().any():
        duplicates = feature_spec_df.loc[feature_spec_df["name"].duplicated(), "name"].tolist()
        raise ValueError(f"Найдены дубли в feature spec: {duplicates}")

    if not final_dataset.empty and final_dataset.columns.duplicated().any():
        duplicates = final_dataset.columns[final_dataset.columns.duplicated()].tolist()
        raise ValueError(f"Найдены дубли колонок в итоговом датасете: {duplicates}")

    if output_dataset_path is not None:
        output_dataset_path = Path(output_dataset_path)
        output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        final_dataset.to_csv(output_dataset_path, index=False)

    if output_feature_spec_path is not None:
        output_feature_spec_path = Path(output_feature_spec_path)
        output_feature_spec_path.parent.mkdir(parents=True, exist_ok=True)
        feature_spec_df.to_csv(output_feature_spec_path, index=False)

    return final_dataset, feature_spec_df
