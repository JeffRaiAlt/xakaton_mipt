from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from feature_audit.selector.models.model_forest import do_work

def main() -> None:

    result = do_work(
        data_path=str(PROJECT_ROOT / "src" / "tmp" / "out" / "final_dataset_reduced.csv"),
        target="buyout_flag",
        time_col="sale_ts",
        random_state=42,
        test_size=0.2,
        n_splits=5,
        n_trials=3,
        min_recall_0=0.90,
        min_precision_1=0.90,
    )
    print("Best params:", result)


if __name__ == "__main__":
    main()