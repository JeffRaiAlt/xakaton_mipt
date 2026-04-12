from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path("../../").resolve()
sys.path.append(str(PROJECT_ROOT))

INPUT_PATH = Path("../../../assembled_outputs"
                  "/final_dataset_from_notebooks.csv")
TMP_DIR = Path("tmp/")
TMP_DIR.mkdir(parents=True, exist_ok=True)

RESULT_OUTPUT_DIR = Path("result/")
RESULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "buyout_flag"
RANDOM_STATE = 42
TEST_SIZE = 0.2

N_SPLITS = 5
N_TRIALS = 1
TOP_N = 50

START_DATE = pd.Timestamp("2025-03-01")
END_DATE = pd.Timestamp("2026-03-29")


df = pd.read_csv(INPUT_PATH, low_memory=False)

df = df[~df[TARGET].isna()].copy()
df[TARGET] = df[TARGET].astype(int)

print("После очистки:", df.shape)
print("Распределение target:")
print(df[TARGET].value_counts(dropna=False))


leakage_or_suspicious_cols = [
    "contact_LTV",
    "has_contact_LTV",
    "contact_loyalty",
    "buyout_flag_lag30",
    "buyout_flag_lag60",
    "buyout_flag_ma30",
    "row_id"
]

df = df.drop(columns=leakage_or_suspicious_cols, errors="ignore")


if "lead_created_dt" in df.columns:
    df["lead_created_dt"] = pd.to_datetime(df["lead_created_dt"], errors="coerce")
    try:
        df["lead_created_dt"] = df["lead_created_dt"].dt.tz_localize(None)
    except TypeError:
        pass

    df = df[df["lead_created_dt"].between(START_DATE, END_DATE)].copy()


drop_cols = [
    "lead_created_dt",
    "lead_utm_id_1",
    "lead_utm_id_2",
    "lead_utm_id_3",
    "lead_utm_position",
    "lead_utm_reatrgeting_id",
    "sale_date",
]

df = df.drop(columns=drop_cols, errors="ignore")


X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

print("\nShape:", X.shape, y.shape)

cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "int32", "float64", "float32", "bool"]).columns.tolist()

print("\nCategorical columns:", len(cat_cols))
print("Numeric columns:", len(num_cols))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


X_train_enc = X_train.copy()
X_test_enc = X_test.copy()

low_card_threshold = 10
encoding_info = []

for col in cat_cols:
    train_values = X_train_enc[col].astype(str)
    test_values = X_test_enc[col].astype(str)

    n_unique = train_values.nunique()

    if n_unique <= low_card_threshold:
        mapping = {v: i for i, v in enumerate(train_values.unique())}
        X_train_enc[col] = train_values.map(mapping).astype(int)
        X_test_enc[col] = test_values.map(mapping).fillna(-1).astype(int)
        encoding_info.append({
            "feature": col,
            "encoding_type": "label_like_integer",
            "n_unique_train": int(n_unique),
        })
    else:
        freq = train_values.value_counts(normalize=True)
        X_train_enc[col] = train_values.map(freq).astype(float)
        X_test_enc[col] = test_values.map(freq).fillna(0.0).astype(float)
        encoding_info.append({
            "feature": col,
            "encoding_type": "frequency",
            "n_unique_train": int(n_unique),
        })

encoding_info_df = pd.DataFrame(encoding_info)

# todo проверить на утечки
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def objective(trial):
    class_weight_key = trial.suggest_categorical(
        "class_weight",
        ["balanced", "balanced_subsample", "none"]
    )

    class_weight_map = {
        "balanced": "balanced",
        "balanced_subsample": "balanced_subsample",
        "none": None,
    }

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 500),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 25),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 12),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": class_weight_map[class_weight_key],
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_enc, y_train)):
        X_tr = X_train_enc.iloc[train_idx]
        X_val = X_train_enc.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)

        p1_val = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, p1_val)
        fold_scores.append(score)

        trial.report(float(np.mean(fold_scores)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=8,
        n_warmup_steps=1,
        interval_steps=1,
    )
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("=" * 60)
print("BEST OPTUNA SCORE (CV ROC-AUC):", study.best_value)
print("BEST PARAMS:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")


best_params = study.best_params.copy()

class_weight_map = {
    "balanced": "balanced",
    "balanced_subsample": "balanced_subsample",
    "none": None,
}

best_params["class_weight"] = class_weight_map[best_params["class_weight"]]
best_params["random_state"] = RANDOM_STATE
best_params["n_jobs"] = -1

best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train_enc, y_train)

train_pred_proba = best_model.predict_proba(X_train_enc)[:, 1]
test_pred_proba = best_model.predict_proba(X_test_enc)[:, 1]

train_roc_auc = roc_auc_score(y_train, train_pred_proba)
test_roc_auc = roc_auc_score(y_test, test_pred_proba)

print("=" * 60)
print(f"Train ROC-AUC: {train_roc_auc:.6f}")
print(f"Test ROC-AUC: {test_roc_auc:.6f}")



feature_importance_df = pd.DataFrame({
    "feature": X_train_enc.columns,
    "importance": best_model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("=" * 60)
print("TOP 30 FEATURES")
print(feature_importance_df.head(30))

selected_features_all = feature_importance_df["feature"].tolist()
selected_features_top = feature_importance_df.head(TOP_N)["feature"].tolist()

print("=" * 60)
print("SELECTED ORIGINAL FEATURES (all ranked):", len(selected_features_all))
print(f"SELECTED TOP {TOP_N} ORIGINAL FEATURES:", len(selected_features_top))


feature_importance_df.to_csv(
    RESULT_OUTPUT_DIR / "best_features_rf_filter.csv",
    index=False,
    encoding="utf-8-sig",
)

feature_importance_df.head(30).to_csv(
    TMP_DIR / "top30_features_rf_filter.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame({"feature": selected_features_all}).to_csv(
    TMP_DIR / "rf_selected_features_all_ranked.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame({"feature": selected_features_top}).to_csv(
    TMP_DIR / "rf_selected_features_top.csv",
    index=False,
    encoding="utf-8-sig",
)

encoding_info_df.to_csv(
    TMP_DIR / "rf_encoding_info.csv",
    index=False,
    encoding="utf-8-sig",
)

with open( TMP_DIR / "best_params_rf_filter.json", "w",
           encoding="utf-8") as f:
    json.dump(study.best_params, f, ensure_ascii=False, indent=4)

n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
n_completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

summary_df = pd.DataFrame([{
    "cv_best_roc_auc": study.best_value,
    "train_roc_auc": train_roc_auc,
    "test_roc_auc": test_roc_auc,
    "n_trials": len(study.trials),
    "n_pruned": n_pruned,
    "n_completed": n_completed,
    "n_selected_original_features": len(selected_features_all),
    "top_n_original_features": TOP_N,
}])

summary_df.to_csv(
    TMP_DIR / "rf_filter_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("=" * 60)
print("FILES SAVED:")