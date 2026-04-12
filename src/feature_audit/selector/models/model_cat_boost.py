from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)


RANDOM_STATE = 42
TARGET_COL = "buyout_flag"
TIME_COL = "sale_ts"

MIN_RECALL_0 = 0.90
MIN_PRECISION_1 = 0.90
THRESHOLDS = np.arange(0.01, 1.00, 0.01)

N_TRIALS = 3
TIMEOUT = None
N_SPLITS = 3

DATA_PATH = "../out/final_dataset_reduced.csv"


df = pd.read_csv(DATA_PATH)
print(f"[INFO] Raw dataset shape: {df.shape}")

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found")
if TIME_COL not in df.columns:
    raise ValueError(f"Time column '{TIME_COL}' not found")

df = df.copy()
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

before_drop = len(df)
df = df.dropna(subset=[TARGET_COL, TIME_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)
df = df.sort_values(TIME_COL).reset_index(drop=True)

print(f"[INFO] Dropped rows with NaN in target/time: {before_drop - len(df)}")
print(f"[INFO] Dataset shape after cleaning: {df.shape}")
print("[INFO] Target distribution:")
print(df[TARGET_COL].value_counts(dropna=False))


def sanitize_categorical_columns(df: pd.DataFrame, target_col: str, time_col: str):
    df = df.copy()

    feature_cols = [c for c in df.columns if c not in [target_col, time_col]]
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        s = df[col].astype("object")
        s = s.where(pd.notna(s), "missing")
        df[col] = s.astype(str)

    return df, cat_cols


def debug_catboost_cats(X: pd.DataFrame, cat_cols: list[str], stage: str):
    bad_cols = []

    for col in cat_cols:
        has_nan = X[col].isna().any()
        mask_bad = ~X[col].map(lambda x: isinstance(x, (str, int, np.integer)))
        has_bad = mask_bad.any()

        if has_nan or has_bad:
            bad_cols.append(col)
            print(f"\n[BAD][{stage}] column={col}")
            print("dtype:", X[col].dtype)
            print("has_nan:", has_nan)
            print("bad examples:")
            print(X.loc[mask_bad | X[col].isna(), col].head(10).tolist())

    if not bad_cols:
        print(f"[INFO] No bad categorical columns at stage: {stage}")



def time_train_valid_test_split(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    train_size: float = 0.64,
    valid_size: float = 0.16,
):
    n = len(df)
    train_end = int(n * train_size)
    valid_end = int(n * (train_size + valid_size))

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        raise ValueError("One of train/valid/test splits is empty. Check dataset size.")

    return train_df, valid_df, test_df


def make_time_folds(df: pd.DataFrame, n_splits: int = 3):
    """
    Простые последовательные фолды без purge/embargo.
    На каждом шаге train = всё прошлое, val = следующий блок.
    """
    n = len(df)
    fold_size = n // (n_splits + 1)
    folds = []

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_end = fold_size * (i + 2) if i < n_splits - 1 else n

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        folds.append((train_idx, val_idx))

    if not folds:
        raise ValueError("No valid time folds were created.")

    return folds


def split_xy(frame: pd.DataFrame):
    X_part = frame.drop(columns=[TARGET_COL, TIME_COL]).copy()
    y_part = frame[TARGET_COL].copy()
    return X_part, y_part


df, cat_features = sanitize_categorical_columns(df, TARGET_COL, TIME_COL)
print(f"[INFO] Number of categorical features after sanitize: {len(cat_features)}")

train_df, valid_df, test_df = time_train_valid_test_split(df, TARGET_COL, TIME_COL)

X_train, y_train = split_xy(train_df)
X_valid, y_valid = split_xy(valid_df)
X_test, y_test = split_xy(test_df)

print("[INFO] Split sizes:")
print(f"  train: {X_train.shape}")
print(f"  valid: {X_valid.shape}")
print(f"  test : {X_test.shape}")
print(
    f"[INFO] Time ranges: "
    f"train=[{train_df[TIME_COL].min()} .. {train_df[TIME_COL].max()}], "
    f"valid=[{valid_df[TIME_COL].min()} .. {valid_df[TIME_COL].max()}], "
    f"test=[{test_df[TIME_COL].min()} .. {test_df[TIME_COL].max()}]"
)

debug_catboost_cats(X_train, cat_features, "train_before_optuna")
debug_catboost_cats(X_valid, cat_features, "valid_before_optuna")
debug_catboost_cats(X_test, cat_features, "test_before_optuna")



def safe_binary_arrays(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_score)

    y_true = y_true[mask].astype(int)
    y_score = y_score[mask]

    if len(y_true) == 0:
        raise ValueError("No valid rows left after NaN filtering in metrics.")

    return y_true, y_score


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 150, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 7),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 10.0, log=True),
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": RANDOM_STATE,
        "verbose": 0,
    }

    folds = make_time_folds(train_df, n_splits=N_SPLITS)
    scores = []

    for fold_num, (train_idx, val_idx) in enumerate(folds, start=1):
        fold_train = train_df.iloc[train_idx].copy()
        fold_val = train_df.iloc[val_idx].copy()

        X_tr, y_tr = split_xy(fold_train)
        X_val, y_val = split_xy(fold_val)

        #debug_catboost_cats(X_tr, cat_features, f"fold_{fold_num}_train")
        #debug_catboost_cats(X_val, cat_features, f"fold_{fold_num}_val")

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=50,
        )

        p1 = model.predict_proba(X_val)[:, 1]
        p0 = 1.0 - p1

        y_val_bin = (y_val == 0).astype(int)
        y_val_bin, p0 = safe_binary_arrays(y_val_bin, p0)
        pr_auc_0 = average_precision_score(y_val_bin, p0)
        scores.append(pr_auc_0)

    return float(np.mean(scores))


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

print("\n" + "=" * 60)
print("BEST OPTUNA RESULT")
print("=" * 60)
print("Best PR-AUC class 0:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")


best_params = study.best_params.copy()
best_params.update(
    {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": RANDOM_STATE,
        "verbose": 100,
    }
)

# финальная модель обучается только на прошлом: train,
# ранняя остановка — по valid,
# test не используется до финальной оценки.
final_model = CatBoostClassifier(**best_params)
final_model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_valid, y_valid),
    use_best_model=True,
    early_stopping_rounds=100,
)


p1_test = final_model.predict_proba(X_test)[:, 1]
p0_test = 1.0 - p1_test

y_test_0 = (y_test == 0).astype(int)
y_test_0, p0_test = safe_binary_arrays(y_test_0, p0_test)
pr_auc_0 = average_precision_score(y_test_0, p0_test)

print("\n" + "=" * 60)
print("FINAL TEST METRICS")
print("=" * 60)
print(f"PR-AUC class 0: {pr_auc_0:.6f}")


def get_metrics_for_threshold(y_true, p1, threshold):
    y_true, p1 = safe_binary_arrays(y_true, p1)
    y_pred = (p1 >= threshold).astype(int)

    precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    precision_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    f2_0 = fbeta_score(y_true, y_pred, beta=2, pos_label=0, zero_division=0)

    return {
        "threshold": threshold,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f2_0": f2_0,
    }


rows = [get_metrics_for_threshold(y_test, p1_test, t) for t in THRESHOLDS]
threshold_df = pd.DataFrame(rows)

filtered_df = threshold_df[
    (threshold_df["recall_0"] >= MIN_RECALL_0)
    & (threshold_df["precision_1"] >= MIN_PRECISION_1)
]

if len(filtered_df) > 0:
    best_row = filtered_df.sort_values("f2_0", ascending=False).iloc[0]
else:
    best_row = threshold_df.sort_values("f2_0", ascending=False).iloc[0]

best_threshold = float(best_row["threshold"])
y_test_clean, p1_test_clean = safe_binary_arrays(y_test, p1_test)
y_pred_best = (p1_test_clean >= best_threshold).astype(int)

print("\nBEST THRESHOLD:", best_threshold)
print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test_clean, y_pred_best))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test_clean, y_pred_best, digits=6))
