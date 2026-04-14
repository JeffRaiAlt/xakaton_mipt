import sys
from pathlib import Path

from feature_audit.selector.dataset_creator import do_work
from feature_audit.selector.strategies.weighted_rank_conflict import WeightedRankConflictStrategy
from feature_audit.selector.feature_selector_fr import prepare_features_fr
from feature_audit.selector.feature_selector_catboost import prepare_features_cat
from feature_audit.selector.feature_selector_logreg import prepare_features_reg

# === базовые директории ===
SRC_DIR = Path(__file__).resolve().parent          # src
PROJECT_ROOT = SRC_DIR.parent                      # project

# чтобы работали импорты feature_audit
sys.path.append(str(SRC_DIR))

# === tmp структура ===
TMP_DIR = SRC_DIR / "tmp"
RESULT_OUTPUT_DIR = TMP_DIR / "result"
TMP_OUTPUT_DIR = TMP_DIR / "out"

RESULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === входной датасет ===
INPUT_PATH = PROJECT_ROOT / "assembled_outputs" / "final_dataset_from_notebooks.csv"


RANDOM_STATE = 42
N_SPLITS = 5
TOP_N = 50
START_DATE = "2025-03-01"
END_DATE = "2026-03-29"

if __name__ == "__main__":

    # Step1 готовим список признаков
    result = prepare_features_fr (
        data_path=str(INPUT_PATH),
        output_dir=str(PROJECT_ROOT / "src" / "tmp" / "result"),
        target="buyout_flag",
        random_state=RANDOM_STATE,
        test_size=0.2,
        n_splits=N_SPLITS,
        n_trials=1,
        top_n=TOP_N,
        start_date=START_DATE,
        end_date=END_DATE,
        date_filter_col="lead_created_dt",
    )

    print("Train ROC-AUC (FR):", result["train_roc_auc"])
    print("Test ROC-AUC (FR):", result["test_roc_auc"])
    print("N selected (FR):", len(result["selected_features"]))

    result = prepare_features_cat (
        data_path=str(INPUT_PATH),
        output_dir=str(PROJECT_ROOT / "src" / "tmp" / "result"),
        target="buyout_flag",
        random_state=RANDOM_STATE,
        test_size=0.2,
        n_splits=N_SPLITS,
        n_trials=1,
        top_n=TOP_N,
        start_date=START_DATE,
        end_date=END_DATE,
        date_filter_col="lead_created_dt",
    )

    print("Train ROC-AUC (CAT):", result["train_roc_auc"])
    print("Test ROC-AUC (CAT):", result["test_roc_auc"])
    print("N selected (CAT):", len(result["selected_features"]))

    result = prepare_features_reg (
        data_path=str(INPUT_PATH),
        output_dir=str(PROJECT_ROOT / "src" / "tmp" / "result"),
        target="buyout_flag",
        random_state=RANDOM_STATE,
        test_size=0.2,
        n_splits=N_SPLITS,
        n_trials=1,
        top_n=TOP_N,
        start_date=START_DATE,
        end_date=END_DATE,
        date_filter_col="lead_created_dt",
    )

    print("Train ROC-AUC (REG):", result["train_roc_auc"])
    print("Test ROC-AUC (REG):", result["test_roc_auc"])
    print("N selected (REG):", len(result["selected_features"]))

    strategy = WeightedRankConflictStrategy(
        logreg_path=RESULT_OUTPUT_DIR / "best_features_logreg_filter.csv",
        rf_path=RESULT_OUTPUT_DIR / "best_features_rf_filter.csv",
        catboost_path=RESULT_OUTPUT_DIR / "best_features_catboost.csv",
        top_k=30,
        final_top_n=40,
        w_logreg=40,
        w_rf=10,
        w_cb=50,
        random_state=RANDOM_STATE,
    )

    do_work(
        strategy,
        data_path=INPUT_PATH,
        output_path=TMP_OUTPUT_DIR / "final_dataset_reduced.csv",
    )


