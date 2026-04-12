from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import optuna

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path("../../").resolve()
sys.path.append(str(PROJECT_ROOT))

INPUT_PATH = Path("../../../assembled_outputs/final_dataset_from_notebooks"
                  ".csv")
TMP_OUTPUT_DIR = Path("tmp/")
TMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_OUTPUT_DIR = Path("result/")
RESULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "buyout_flag"
RANDOM_STATE = 42
TEST_SIZE = 0.2

N_SPLITS = 5
N_TRIALS = 1
TOP_N = 50
TOP_K_PER_FOLD = 30

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


def preprocess_for_catboost(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    datetime_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()

    for col in datetime_cols:
        dt = df[col].dt
        df[f"{col}_year"] = dt.year
        df[f"{col}_month"] = dt.month
        df[f"{col}_day"] = dt.day
        df[f"{col}_weekday"] = dt.weekday
        df[f"{col}_hour"] = dt.hour

    df = df.drop(columns=datetime_cols)
    return df


def prepare_catboost_data(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train = preprocess_for_catboost(X_train)
    X_test = preprocess_for_catboost(X_test)

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [col for col in X_train.columns if col not in cat_cols]

    for col in cat_cols:
        X_train[col] = X_train[col].where(~X_train[col].isna(), "missing").astype(str)
        X_test[col] = X_test[col].where(~X_test[col].isna(), "missing").astype(str)

    for col in num_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    y_train = pd.Series(y_train).astype(int)
    y_test = pd.Series(y_test).astype(int)

    return X_train, X_test, y_train, y_test, cat_cols, num_cols


X_train, X_test, y_train, y_test, cat_cols, num_cols = prepare_catboost_data(df, TARGET)

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Categorical features:", len(cat_cols))
print("Numerical features:", len(num_cols))


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 900),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights",
            ["Balanced", "SqrtBalanced"]
        ),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
        "random_seed": RANDOM_STATE,
    }

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx].copy()
        X_val = X_train.iloc[val_idx].copy()
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model = CatBoostClassifier(**params)

        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=50,
        )

        p1_val = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, p1_val)
        fold_scores.append(fold_auc)

        trial.report(float(np.mean(fold_scores)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
        interval_steps=1,
    ),
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("=" * 60)
print("BEST OPTUNA SCORE (CV ROC-AUC):", study.best_value)
print("BEST PARAMS:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")


best_params = study.best_params.copy()
best_params.update({
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": 100,
    "random_seed": RANDOM_STATE,
})

best_model = CatBoostClassifier(**best_params)

best_model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
)

train_pred_proba = best_model.predict_proba(X_train)[:, 1]
test_pred_proba = best_model.predict_proba(X_test)[:, 1]

train_roc_auc = roc_auc_score(y_train, train_pred_proba)
test_roc_auc = roc_auc_score(y_test, test_pred_proba)
gini = 2 * test_roc_auc - 1

print("=" * 60)
print(f"Train ROC-AUC: {train_roc_auc:.6f}")
print(f"Test ROC-AUC: {test_roc_auc:.6f}")
print(f"GINI: {gini:.6f}")


fold_importance_list = []
fold_top_features_list = []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.iloc[train_idx].copy()
    X_val = X_train.iloc[val_idx].copy()
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    model_fold = CatBoostClassifier(**best_params)

    model_fold.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=50,
        verbose=0,
    )

    fold_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model_fold.get_feature_importance()
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fold_imp["fold"] = fold_idx
    fold_imp["rank_in_fold"] = np.arange(1, len(fold_imp) + 1)

    fold_importance_list.append(fold_imp)
    fold_top_features_list.append(
        fold_imp.head(TOP_K_PER_FOLD)[["feature"]].assign(is_top_k=1, fold=fold_idx)
    )


final_feature_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "final_importance": best_model.get_feature_importance()
}).sort_values("final_importance", ascending=False).reset_index(drop=True)

all_fold_importance_df = pd.concat(fold_importance_list, axis=0, ignore_index=True)
all_fold_top_df = pd.concat(fold_top_features_list, axis=0, ignore_index=True)

fold_stats_df = (
    all_fold_importance_df
    .groupby("feature", as_index=False)
    .agg(
        mean_importance_cv=("importance", "mean"),
        std_importance_cv=("importance", "std"),
        min_rank_cv=("rank_in_fold", "min"),
        mean_rank_cv=("rank_in_fold", "mean"),
    )
)

top_freq_df = (
    all_fold_top_df
    .groupby("feature", as_index=False)
    .agg(top_k_hits=("is_top_k", "sum"))
)

top_freq_df["top_k_share"] = top_freq_df["top_k_hits"] / N_SPLITS

feature_importance_df = (
    final_feature_importance_df
    .merge(fold_stats_df, on="feature", how="left")
    .merge(top_freq_df, on="feature", how="left")
)

feature_importance_df["top_k_hits"] = feature_importance_df["top_k_hits"].fillna(0).astype(int)
feature_importance_df["top_k_share"] = feature_importance_df["top_k_share"].fillna(0.0)
feature_importance_df["std_importance_cv"] = feature_importance_df["std_importance_cv"].fillna(0.0)


def safe_minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.nunique(dropna=False) <= 1:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


feature_importance_df["final_importance_norm"] = safe_minmax(feature_importance_df["final_importance"])
feature_importance_df["mean_importance_cv_norm"] = safe_minmax(feature_importance_df["mean_importance_cv"])
feature_importance_df["top_k_share_norm"] = safe_minmax(feature_importance_df["top_k_share"])

feature_importance_df["stability_penalty"] = 1.0 / (1.0 + feature_importance_df["std_importance_cv"])

feature_importance_df["robust_score"] = (
    0.45 * feature_importance_df["final_importance_norm"] +
    0.35 * feature_importance_df["mean_importance_cv_norm"] +
    0.20 * feature_importance_df["top_k_share_norm"]
) * feature_importance_df["stability_penalty"]

feature_importance_df = feature_importance_df.sort_values(
    by=["robust_score", "top_k_hits", "final_importance"],
    ascending=[False, False, False]
).reset_index(drop=True)

interesting_features_df = feature_importance_df[
    (
        (feature_importance_df["top_k_hits"] >= max(2, N_SPLITS // 2 + 1)) |
        (feature_importance_df["final_importance"] > 0)
    )
].copy().head(TOP_N).reset_index(drop=True)

interesting_feature_list = interesting_features_df["feature"].tolist()

print("=" * 60)
print("TOP FEATURES")
print(feature_importance_df.head(30)[[
    "feature",
    "robust_score",
    "final_importance",
    "mean_importance_cv",
    "std_importance_cv",
    "top_k_hits",
    "top_k_share",
    "mean_rank_cv",
]])

print("=" * 60)
print(f"INTERESTING FEATURES ({len(interesting_feature_list)}):")
print(interesting_feature_list)


feature_importance_df.to_csv(
    RESULT_OUTPUT_DIR / "best_features_catboost.csv", index=False,
    encoding="utf-8-sig")
feature_importance_df.head(30).to_csv(TMP_OUTPUT_DIR /
                                      "top30_features_catboost.csv", index=False, encoding="utf-8-sig")
interesting_features_df.to_csv(TMP_OUTPUT_DIR /
                               "interesting_features_catboost.csv", index=False, encoding="utf-8-sig")
all_fold_importance_df.to_csv(TMP_OUTPUT_DIR /
                              "catboost_fold_importances.csv", index=False, encoding="utf-8-sig")

with open(TMP_OUTPUT_DIR /
          "interesting_feature_list_catboost.json", "w", encoding="utf-8") as f:
    json.dump(interesting_feature_list, f, ensure_ascii=False, indent=4)

with open(TMP_OUTPUT_DIR / "best_params_catboost.json", "w",
          encoding="utf-8") as f:
    json.dump(study.best_params, f, ensure_ascii=False, indent=4)

summary_df = pd.DataFrame([{
    "train_roc_auc": train_roc_auc,
    "test_roc_auc": test_roc_auc,
    "gini": gini,
    "n_features_total": X_train.shape[1],
    "n_interesting_features": len(interesting_feature_list),
    "n_trials": len(study.trials),
    "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
    "n_completed": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
    "top_n": TOP_N,
    "top_k_per_fold": TOP_K_PER_FOLD,
}])

summary_df.to_csv(TMP_OUTPUT_DIR / "catboost_summary_metrics.csv",
                  index=False, encoding="utf-8-sig")

print("=" * 60)
print("FILES SAVED:")