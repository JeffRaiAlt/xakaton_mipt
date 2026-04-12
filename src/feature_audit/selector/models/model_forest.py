import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
)

# ======================================================
# SETTINGS
# ======================================================
RANDOM_STATE = 42

TARGET = "buyout_flag"
TIME_COL = "sale_ts"

DATA_PATH = "../out/final_dataset_reduced.csv"

TEST_SIZE = 0.2
N_SPLITS = 5
N_TRIALS = 30

THRESHOLDS = np.arange(0.05, 0.96, 0.01)
MIN_RECALL_0 = 0.90
MIN_PRECISION_1 = 0.90

# ======================================================
# LOAD
# ======================================================
df = pd.read_csv(DATA_PATH, low_memory=False)

df = df[~df[TARGET].isna()].copy()
df[TARGET] = df[TARGET].astype(int)

df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL]).copy()

df = df.sort_values(TIME_COL).reset_index(drop=True)

print("DATA SHAPE:", df.shape)
print("TARGET DISTRIBUTION:")
print(df[TARGET].value_counts(dropna=False))

# ======================================================
# SPLIT (TIME-BASED)
# ======================================================
split_idx = int(len(df) * (1 - TEST_SIZE))

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

features = [c for c in df.columns if c not in [TARGET, TIME_COL]]

X_train = train_df[features].copy()
y_train = train_df[TARGET].copy()

X_test = test_df[features].copy()
y_test = test_df[TARGET].copy()

print("\nTRAIN PERIOD:", train_df[TIME_COL].min(), "->", train_df[TIME_COL].max())
print("TEST  PERIOD:", test_df[TIME_COL].min(), "->", test_df[TIME_COL].max())
print("N FEATURES:", len(features))

# ======================================================
# PREPROCESS HELPERS
# ======================================================
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

print("CATEGORICAL COLS:", len(cat_cols))
print("NUMERICAL COLS:", len(num_cols))


def preprocess_fit_transform(X_tr: pd.DataFrame):
    X_tr = X_tr.copy()

    # categorical -> safe string
    for col in cat_cols:
        s = X_tr[col].astype("object")
        s = s.where(pd.notna(s), "missing")
        X_tr[col] = s.astype(str)

    # numeric NaN -> train medians
    medians = X_tr[num_cols].median()
    if len(num_cols) > 0:
        X_tr[num_cols] = X_tr[num_cols].fillna(medians)

    # frequency encoding ONLY on current train
    freq_maps = {}
    for col in cat_cols:
        freq = X_tr[col].value_counts(normalize=True)
        freq_maps[col] = freq
        X_tr[col] = X_tr[col].map(freq).fillna(0.0)

    return X_tr, medians, freq_maps


def preprocess_transform(X_val: pd.DataFrame, medians, freq_maps):
    X_val = X_val.copy()

    for col in cat_cols:
        s = X_val[col].astype("object")
        s = s.where(pd.notna(s), "missing")
        X_val[col] = s.astype(str)

    if len(num_cols) > 0:
        X_val[num_cols] = X_val[num_cols].fillna(medians)

    for col in cat_cols:
        X_val[col] = X_val[col].map(freq_maps[col]).fillna(0.0)

    return X_val


# fit preprocessing on full train only, then apply to final test
X_train, global_medians, global_freq_maps = preprocess_fit_transform(X_train)
X_test = preprocess_transform(X_test, global_medians, global_freq_maps)

# ======================================================
# TIME FOLDS
# ======================================================
def generate_time_folds(times, n_splits=5):
    times = pd.Series(times).reset_index(drop=True)
    indices = np.arange(len(times))

    fold_size = len(times) // n_splits
    folds = []

    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else len(times)

        val_idx = indices[start:end]
        train_idx = indices[:start]

        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))

    return folds


folds = generate_time_folds(train_df[TIME_COL], N_SPLITS)
print("N FOLDS:", len(folds))

# ======================================================
# SAFE METRIC HELPERS
# ======================================================
def safe_arrays(y, p0, p1=None):
    y = pd.Series(y).reset_index(drop=True)
    p0 = pd.Series(p0).reset_index(drop=True)

    mask = pd.notna(y) & pd.notna(p0) & np.isfinite(p0)

    if p1 is not None:
        p1 = pd.Series(p1).reset_index(drop=True)
        mask = mask & pd.notna(p1) & np.isfinite(p1)
        return y[mask], p0[mask], p1[mask]

    return y[mask], p0[mask]


# ======================================================
# METRICS
# ======================================================
def base_metrics(y, p0, p1):
    y, p0, p1 = safe_arrays(y, p0, p1)
    y0 = (y == 0).astype(int)

    return {
        "pr_auc_0": average_precision_score(y0, p0),
        "roc_auc_1": roc_auc_score(y, p1),
    }


def threshold_metrics(y, p0, thr):
    y, p0 = safe_arrays(y, p0)

    pred = np.where(p0 >= thr, 0, 1)

    return {
        "thr": thr,
        "recall_0": recall_score(y, pred, pos_label=0, zero_division=0),
        "precision_0": precision_score(y, pred, pos_label=0, zero_division=0),
        "f2_0": fbeta_score(y, pred, beta=2, pos_label=0, zero_division=0),
        "recall_1": recall_score(y, pred, pos_label=1, zero_division=0),
        "precision_1": precision_score(y, pred, pos_label=1, zero_division=0),
    }


def find_best_threshold(y, p0):
    rows = [threshold_metrics(y, p0, t) for t in THRESHOLDS]
    df_thr = pd.DataFrame(rows)

    filt = df_thr[
        (df_thr["recall_0"] >= MIN_RECALL_0) &
        (df_thr["precision_1"] >= MIN_PRECISION_1)
    ]

    if len(filt) > 0:
        best = filt.sort_values("f2_0", ascending=False).iloc[0]
    else:
        best = df_thr.sort_values("f2_0", ascending=False).iloc[0]

    return float(best["thr"]), df_thr


def eval_thr(y, p0, thr):
    y, p0 = safe_arrays(y, p0)
    pred = np.where(p0 >= thr, 0, 1)
    return y, pred


# ======================================================
# OPTUNA
# ======================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    scores = []

    for tr_idx, val_idx in folds:
        X_tr_raw = train_df.iloc[tr_idx][features].copy()
        y_tr = train_df.iloc[tr_idx][TARGET].copy()

        X_val_raw = train_df.iloc[val_idx][features].copy()
        y_val = train_df.iloc[val_idx][TARGET].copy()

        X_tr, medians, freq_maps = preprocess_fit_transform(X_tr_raw)
        X_val = preprocess_transform(X_val_raw, medians, freq_maps)

        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)

        p1 = model.predict_proba(X_val)[:, 1]
        p0 = 1 - p1

        y_val, p0 = safe_arrays(y_val, p0)
        y0 = (y_val == 0).astype(int)

        score = average_precision_score(y0, p0)
        scores.append(score)

    return float(np.mean(scores))


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params.copy()
best_params["random_state"] = RANDOM_STATE
best_params["n_jobs"] = -1

print("\nBEST OPTUNA RESULT")
print("Best value:", study.best_value)
print("Best params:", best_params)

# ======================================================
# OOF PREDICTIONS FOR THRESHOLD SELECTION
# ======================================================
def get_oof_p0(train_df, features, folds, params):
    oof_p0 = pd.Series(index=train_df.index, dtype=float)

    for tr_idx, val_idx in folds:
        X_tr_raw = train_df.iloc[tr_idx][features].copy()
        y_tr = train_df.iloc[tr_idx][TARGET].copy()

        X_val_raw = train_df.iloc[val_idx][features].copy()

        X_tr, medians, freq_maps = preprocess_fit_transform(X_tr_raw)
        X_val = preprocess_transform(X_val_raw, medians, freq_maps)

        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)

        p1_val = model.predict_proba(X_val)[:, 1]
        p0_val = 1 - p1_val

        oof_p0.iloc[val_idx] = p0_val

    mask = oof_p0.notna()
    return oof_p0, mask


oof_p0, oof_mask = get_oof_p0(train_df, features, folds, best_params)
best_thr, threshold_df = find_best_threshold(y_train.loc[oof_mask], oof_p0.loc[oof_mask])

print("\nBEST THRESHOLD:", best_thr)

# ======================================================
# FINAL MODEL
# ======================================================
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

p1_train = model.predict_proba(X_train)[:, 1]
p0_train = 1 - p1_train

p1_test = model.predict_proba(X_test)[:, 1]
p0_test = 1 - p1_test

# ======================================================
# BASE METRICS
# ======================================================
train_base = base_metrics(y_train, p0_train, p1_train)
test_base = base_metrics(y_test, p0_test, p1_test)

print("\nBASE METRICS")
print("TRAIN PR-AUC 0:", train_base["pr_auc_0"])
print("TEST  PR-AUC 0:", test_base["pr_auc_0"])
print("TRAIN ROC-AUC 1:", train_base["roc_auc_1"])
print("TEST  ROC-AUC 1:", test_base["roc_auc_1"])

# ======================================================
# THRESHOLD EVAL
# ======================================================
y_train_eval, pred_train = eval_thr(y_train, p0_train, best_thr)
y_test_eval, pred_test = eval_thr(y_test, p0_test, best_thr)

print("\nTRAIN")
print(confusion_matrix(y_train_eval, pred_train))
print(classification_report(y_train_eval, pred_train, digits=4))

print("\nTEST")
print(confusion_matrix(y_test_eval, pred_test))
print(classification_report(y_test_eval, pred_test, digits=4))

# ======================================================
# FEATURE IMPORTANCE
# ======================================================
imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTOP FEATURES")
print(imp.head(30))