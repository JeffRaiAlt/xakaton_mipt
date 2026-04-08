from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def load_notebook_outputs(
    block_dirs: Iterable[str | Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    blocks = []
    #specs = []

    for block_dir in block_dirs:
        block_dir = Path(block_dir)

        x_files = sorted(block_dir.glob("*.csv"))

        print(block_dir)
        print("x_files:", x_files)

        if x_files:
            group_df = None

            for x_path in x_files:
                df = pd.read_csv(x_path)

                if "row_id" not in df.columns:
                    raise ValueError(f"{x_path} не содержит row_id")

                if group_df is None:
                    group_df = df
                else:
                    group_df = group_df.merge(df, on="row_id", how="inner")

            blocks.append(group_df)


    if not blocks:
        raise ValueError("Не найдено ни одного X_block*.csv")

    final_dataset = blocks[0]
    for df in blocks[1:]:
        final_dataset = final_dataset.merge(df, on="row_id", how="inner")


    if final_dataset.columns.duplicated().any():
        duplicates = final_dataset.columns[final_dataset.columns.duplicated()].tolist()
        raise ValueError(f"Дубли колонок: {duplicates}")

    return final_dataset, pd.DataFrame()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    block_dirs = [
        project_root / "notebook_outputs" / "group_1",
        project_root / "notebook_outputs" / "group_2",
        project_root / "notebook_outputs" / "group_3",
        project_root / "notebook_outputs" / "group_4",
        project_root / "notebook_outputs" / "group_5",
    ]

    final_dataset, _ = load_notebook_outputs(block_dirs=block_dirs)

    output_dir = project_root / "assembled_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_dataset.to_csv(output_dir / "final_dataset_from_notebooks.csv", index=False)
