from pathlib import Path
import sys
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from feature_audit.selector.models.model_forest import do_work
from feature_audit.selector.models.model_cat_boost import do_work_cat


def save_result(result: dict, base_path: str):
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # отдельно importance
    fi = result.pop("feature_importance")
    fi.to_csv(base_path / "feature_importance.csv", index=False)

    # остальное в json
    with open(base_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)



def main() -> None:
    result = do_work_cat(
        data_path=str(PROJECT_ROOT / "data" / "cleaned_dataset" / "final_dataset.csv"),
        target="buyout_flag",
        time_col="sale_ts",
        test_size=0.2,
        n_splits=5,
        n_trials=7,
        output_path=str(PROJECT_ROOT / "src" / "tmp" / "out")
    )
    save_result(result, str(PROJECT_ROOT / "src"/ "tmp"))
    print("Best params:", result)


if __name__ == "__main__":
    main()