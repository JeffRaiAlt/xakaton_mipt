from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd
import optuna

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    fbeta_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.exceptions import ConvergenceWarning

#todo не готова!!!!
PROJECT_ROOT = Path("../../../../").resolve()
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = "../out/final_dataset_reduced.csv"
REPORT_DIR = Path("tmp/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
TARGET = "buyout_flag"
TIME_COL = "sale_ts"

TEST_SIZE = 0.2
VALID_SIZE = 0.2   # доля от train_valid части
N_SPLITS = 3
N_TRIALS = 25

THRESHOLDS = np.arange(0.05, 0.96, 0.01)


df = pd.read_csv(DATA_PATH, low_memory=False)

df = df[~df[TARGET].isna()].copy()
df[TARGET] = df[TARGET].astype(int)

df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL]).copy()

df = df.sort_values(TIME_COL).reset_index(drop=True)

print("После очистки:", df.shape)
print("Распределение target:")
print(df[TARGET].value_counts(dropna=False))


n = len(df)
test_start = int(n * (1 - TEST_SIZE))

train_valid_df = df.iloc[:test_start].copy()
test_df = df.iloc[test_start:].copy()

valid_start_inside_train = int(len(train_valid_df) * (1 - VALID_SIZE))
train_df = train_valid_df.iloc[:valid_start_inside_train].copy()
valid_df = train_valid_df.iloc[valid_start_inside_train:].copy()

print(
    f"[INFO] Time ranges: "
    f"train=[{train_df[TIME_COL].min()} .. {train_df[TIME_COL].max()}], "
    f"valid=[{valid_df[TIME_COL].min()} .. {valid_df[TIME_COL].max()}], "
    f"test=[{test_df[TIME_COL].min()} .. {test_df[TIME_COL].max()}]"
)


feature_cols = [c for c in df.columns if c not in [TARGET, TIME_COL]]

X_train = train_df[feature_cols].copy()
y_train = train_df[TARGET].copy()

X_valid = valid_df[feature_cols].copy()
y_valid = valid_df[TARGET].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df[TARGET].copy()

print("Train shape:", X_train.shape)
print("Valid shape:", X_valid.shape)
print("Test shape:", X_test.shape)

cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(include=["int64", "int32", "float64", "float32", "bool"]).columns.tolist()

print("\nCategorical columns:", len(cat_cols))
print("Numeric columns:", len(num_cols))


cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols),
])


class_weight_map = {
    "balanced": "balanced",
    "w_3_1": {0: 3, 1: 1},
    "w_2_1": {0: 2, 1: 1},
}


def make_model(C, l1_ratio, class_weight_key, tol):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            class_weight=class_weight_map[class_weight_key],
            tol=tol,
            max_iter=3000,
            random_state=RANDOM_STATE,
        ))
    ])


def build_threshold_df(y_true, p0):
    rows = []

    for thr in THRESHOLDS:
        y_pred = np.where(p0 >= thr, 0, 1)

        rows.append({
            "threshold_0": thr,
            "recall_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            "precision_0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            "f2_0": fbeta_score(y_true, y_pred, beta=2, pos_label=0, zero_division=0),
            "recall_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "precision_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        })

    threshold_df = pd.DataFrame(rows).sort_values(
        by=["f2_0", "recall_0", "precision_0"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return threshold_df



def objective(trial):
    C = trial.suggest_float("C", 1e-3, 3.0, log=True)
    l1_ratio = trial.suggest_categorical("l1_ratio", [0.5, 0.7, 0.9, 1.0])
    class_weight_key = trial.suggest_categorical(
        "class_weight",
        ["balanced", "w_3_1", "w_2_1"]
    )
    tol = trial.suggest_categorical("tol", [1e-3, 3e-3])

    model = make_model(C, l1_ratio, class_weight_key, tol)

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_tr, y_tr)

        p1_val = model.predict_proba(X_val)[:, 1]
        p0_val = 1.0 - p1_val

        thresholds = np.arange(0.10, 0.96, 0.05)
        best_fold_f2_0 = -1.0

        for thr in thresholds:
            y_pred_val = np.where(p0_val >= thr, 0, 1)

            f2_0 = fbeta_score(
                y_val,
                y_pred_val,
                beta=2,
                pos_label=0,
                zero_division=0
            )

            if f2_0 > best_fold_f2_0:
                best_fold_f2_0 = f2_0

        fold_scores.append(best_fold_f2_0)

        mean_score_so_far = float(np.mean(fold_scores))
        trial.report(mean_score_so_far, step=fold_idx)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))



study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
        interval_steps=1
    )
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("=" * 60)
print("BEST OPTUNA SCORE (CV F2 for class 0):", study.best_value)
print("BEST PARAMS:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")

best_params = study.best_params.copy()


best_model = make_model(
    C=best_params["C"],
    l1_ratio=best_params["l1_ratio"],
    class_weight_key=best_params["class_weight"],
    tol=best_params["tol"],
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    best_model.fit(X_train, y_train)

# valid predictions -> choose threshold here
p1_valid = best_model.predict_proba(X_valid)[:, 1]
p0_valid = 1.0 - p1_valid

valid_threshold_df = build_threshold_df(y_valid, p0_valid)
best_threshold_0 = float(valid_threshold_df.loc[0, "threshold_0"])

print("=" * 60)
print("BEST THRESHOLD FOR CLASS 0 (from VALID):", best_threshold_0)
print(valid_threshold_df.head(10))


p1_test = best_model.predict_proba(X_test)[:, 1]
p0_test = 1.0 - p1_test

roc_auc = roc_auc_score(y_test, p1_test)

y_pred_best = np.where(p0_test >= best_threshold_0, 0, 1)

cm = confusion_matrix(y_test, y_pred_best)

recall_0 = recall_score(y_test, y_pred_best, pos_label=0, zero_division=0)
precision_0 = precision_score(y_test, y_pred_best, pos_label=0, zero_division=0)
f2_0 = fbeta_score(y_test, y_pred_best, beta=2, pos_label=0, zero_division=0)

recall_1 = recall_score(y_test, y_pred_best, pos_label=1, zero_division=0)
precision_1 = precision_score(y_test, y_pred_best, pos_label=1, zero_division=0)
f2_1 = fbeta_score(y_test, y_pred_best, beta=2, pos_label=1, zero_division=0)

print("=" * 60)
print("FINAL TEST METRICS AT VALID-SELECTED THRESHOLD")
print("=" * 60)
print("ROC-AUC:", roc_auc)
print("Best threshold for class 0:", best_threshold_0)
print("Recall class 0:", recall_0)
print("Precision class 0:", precision_0)
print("F2 class 0:", f2_0)
print("Recall class 1:", recall_1)
print("Precision class 1:", precision_1)
print("F2 class 1:", f2_1)
print("\nConfusion matrix:")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred_best, digits=4, zero_division=0))


feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
coefs = best_model.named_steps["classifier"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs,
    "abs_coef": np.abs(coefs)
}).sort_values("abs_coef", ascending=False).reset_index(drop=True)

selected_coef_df = coef_df[coef_df["coef"] != 0].copy().reset_index(drop=True)

print("=" * 60)
print("TOP 30 FEATURES BY |COEF|")
print(coef_df.head(30))

print("=" * 60)
print("SELECTED FEATURES (coef != 0):", len(selected_coef_df))


coef_df.to_csv("best_features_logreg.csv", index=False, encoding="utf-8-sig")
selected_coef_df.to_csv("selected_features_logreg.csv", index=False, encoding="utf-8-sig")
valid_threshold_df.to_csv("threshold_metrics_logreg_valid.csv", index=False, encoding="utf-8-sig")

with open(REPORT_DIR / "best_params_logreg.json", "w", encoding="utf-8") as f:
    json.dump(study.best_params, f, ensure_ascii=False, indent=4)

summary_df = pd.DataFrame([{
    "roc_auc": roc_auc,
    "best_threshold_0": best_threshold_0,
    "recall_0": recall_0,
    "precision_0": precision_0,
    "f2_0": f2_0,
    "recall_1": recall_1,
    "precision_1": precision_1,
    "f2_1": f2_1,
    "tn": cm[0, 0],
    "fp": cm[0, 1],
    "fn": cm[1, 0],
    "tp": cm[1, 1],
    "n_trials": len(study.trials),
    "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
    "n_completed": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
    "n_selected_features": len(selected_coef_df),
    "train_start": str(train_df[TIME_COL].min()),
    "train_end": str(train_df[TIME_COL].max()),
    "valid_start": str(valid_df[TIME_COL].min()),
    "valid_end": str(valid_df[TIME_COL].max()),
    "test_start": str(test_df[TIME_COL].min()),
    "test_end": str(test_df[TIME_COL].max()),
}])

summary_df.to_csv("logreg_summary_metrics.csv", index=False, encoding="utf-8-sig")

print("=" * 60)
print("Artifacts saved.")