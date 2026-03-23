from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.spec_converter import convert_feature_spec_csv_to_json


def load_notebook_outputs(
    block_dirs: Iterable[str | Path],
    features_root: str | Path | None = None,
    convert_specs_to_json: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает результаты, сохраненные из исследовательских ноутбуков.

    Для каждой папки ожидаются:
    - X_block.csv
    - feature_spec.csv

    Дополнительно:
    - если convert_specs_to_json=True, то для каждой группы автоматически
      создается/обновляется JSON-файл:
      src/features/group_X/example_feature_spec.json

    Parameters
    ----------
    block_dirs:
        Список директорий вида notebook_outputs/group_X
    features_root:
        Корень папки src/features. Если None, JSON не сохраняются автоматически,
        кроме случая запуска через __main__, где путь вычисляется от проекта.
    convert_specs_to_json:
        Нужно ли автоматически конвертировать feature_spec.csv -> JSON

    Returns
    -------
    final_dataset, final_feature_spec
    """
    blocks = []
    specs = []

    features_root = Path(features_root) if features_root is not None else None

    for block_dir in block_dirs:
        block_dir = Path(block_dir)

        x_path = block_dir / "X_block.csv"
        spec_path = block_dir / "feature_spec.csv"

        if not x_path.exists():
            raise FileNotFoundError(f"Не найден файл: {x_path}")
        if not spec_path.exists():
            raise FileNotFoundError(f"Не найден файл: {spec_path}")

        X_block = pd.read_csv(x_path)
        feature_spec = pd.read_csv(spec_path)

        # group name = имя папки notebook_outputs/group_X -> group_X
        group_name = block_dir.name

        if convert_specs_to_json and features_root is not None:
            json_path = features_root / group_name / "example_feature_spec.json"
            convert_feature_spec_csv_to_json(
                csv_path=spec_path,
                json_path=json_path,
            )

        blocks.append(X_block)
        specs.append(feature_spec)

    final_dataset = pd.concat(blocks, axis=1) if blocks else pd.DataFrame()
    final_feature_spec = pd.concat(specs, axis=0, ignore_index=True) if specs else pd.DataFrame()

    if not final_dataset.empty and final_dataset.columns.duplicated().any():
        duplicates = final_dataset.columns[final_dataset.columns.duplicated()].tolist()
        raise ValueError(f"Найдены дубли колонок в итоговом датасете: {duplicates}")

    if not final_feature_spec.empty and final_feature_spec["name"].duplicated().any():
        duplicates = final_feature_spec.loc[
            final_feature_spec["name"].duplicated(), "name"
        ].tolist()
        raise ValueError(f"Найдены дубли имен в feature_spec: {duplicates}")

    return final_dataset, final_feature_spec


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    block_dirs = [
        project_root / "notebook_outputs" / "group_1",
        project_root / "notebook_outputs" / "group_2",
        project_root / "notebook_outputs" / "group_3",
        project_root / "notebook_outputs" / "group_4",
        project_root / "notebook_outputs" / "group_5",
    ]

    final_dataset, final_feature_spec = load_notebook_outputs(
        block_dirs=block_dirs,
        features_root=project_root / "src" / "features",
        convert_specs_to_json=True,
    )

    output_dir = project_root / "assembled_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_dataset.to_csv(output_dir / "final_dataset_from_notebooks.csv", index=False)
    final_feature_spec.to_csv(output_dir / "feature_spec_from_notebooks.csv", index=False)

    print("Saved dataset:", output_dir / "final_dataset_from_notebooks.csv")
    print("Saved feature spec:", output_dir / "feature_spec_from_notebooks.csv")
    print(final_dataset.shape)
    print(final_feature_spec.shape)
