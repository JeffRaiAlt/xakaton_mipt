from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning


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

N_SPLITS = 3
N_TRIALS = 1
TOP_N = 50

START_DATE = pd.Timestamp("2025-03-01")
END_DATE = pd.Timestamp("2026-03-29")


df = pd.read_csv(INPUT_PATH, low_memory=False)

# очистка таргета
df = df[~df[TARGET].isna()].copy()
df[TARGET] = df[TARGET].astype(int)

print("После очистки:", df.shape)
print("Распределение target:")
print(df[TARGET].value_counts(dropna=False))


id_like_cols = ["row_id"]

leakage_or_suspicious_cols = [
    "contact_LTV",
    "has_contact_LTV",
    "contact_loyalty",
    "buyout_flag_lag30",
    "buyout_flag_lag60",
    "buyout_flag_ma30",
]

df = df.drop(columns=leakage_or_suspicious_cols, errors="ignore")
df = df.drop(columns=id_like_cols, errors="ignore")


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


num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols),
])

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


cv_splits = list(cv.split(X_train, y_train))

preprocessor_fitted = clone(preprocessor)
X_train_prepared = preprocessor_fitted.fit_transform(X_train, y_train)

# Если y_train pandas Series, переведем в numpy для более быстрого индексирования
y_train_array = y_train.to_numpy()

def objective(trial):
    C = trial.suggest_float("C", 1e-2, 3.0, log=True)
    l1_ratio = trial.suggest_categorical("l1_ratio", [0.5, 0.7, 0.9, 1.0])
    tol = trial.suggest_categorical("tol", [1e-3, 3e-3])

    clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=C,
        l1_ratio=l1_ratio,
        tol=tol,
        max_iter=2000,      # было 5000, обычно можно сильно снизить
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    fold_scores = []
    had_convergence_warning = False

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_tr = X_train_prepared[train_idx]
        X_val = X_train_prepared[val_idx]
        y_tr = y_train_array[train_idx]
        y_val = y_train_array[val_idx]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=ConvergenceWarning)
            clf.fit(X_tr, y_tr)

            if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                had_convergence_warning = True

        p1_val = clf.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, p1_val)
        fold_scores.append(score)

        trial.report(float(np.mean(fold_scores)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("had_convergence_warning", had_convergence_warning)
    return float(np.mean(fold_scores))


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
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

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=best_params["C"],
        l1_ratio=best_params["l1_ratio"],
        tol=best_params["tol"],
        max_iter=5000,
        class_weight=None,
        random_state=RANDOM_STATE,
    ))
])

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", category=ConvergenceWarning)
    best_model.fit(X_train, y_train)
    best_model_had_convergence_warning = any(
        issubclass(warn.category, ConvergenceWarning) for warn in w
    )

train_pred_proba = best_model.predict_proba(X_train)[:, 1]
test_pred_proba = best_model.predict_proba(X_test)[:, 1]

train_roc_auc = roc_auc_score(y_train, train_pred_proba)
test_roc_auc = roc_auc_score(y_test, test_pred_proba)

print("=" * 60)
print(f"Train ROC-AUC: {train_roc_auc:.6f}")
print(f"Test ROC-AUC: {test_roc_auc:.6f}")
print(f"Convergence warning on best model: {best_model_had_convergence_warning}")


feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
coefs = best_model.named_steps["classifier"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs,
    "abs_coef": np.abs(coefs),
}).sort_values("abs_coef", ascending=False).reset_index(drop=True)

selected_coef_df = coef_df[coef_df["coef"] != 0].copy().reset_index(drop=True)

print("=" * 60)
print("TOP 30 FEATURES BY |COEF|")
print(coef_df.head(30))

print("=" * 60)
print("SELECTED ONE-HOT / NUM FEATURES (coef != 0):", len(selected_coef_df))


def extract_original_feature(feature_name: str, num_cols_: list[str], cat_cols_: list[str]) -> str:
    if feature_name.startswith("num__"):
        return feature_name.replace("num__", "", 1)

    if feature_name.startswith("cat__"):
        base = feature_name.replace("cat__", "", 1)

        # Ищем исходный categorical column как самый длинный совпавший префикс
        matches = [col for col in cat_cols_ if base == col or base.startswith(col + "_")]
        if matches:
            return max(matches, key=len)

        return base

    return feature_name


selected_coef_df["base_feature"] = selected_coef_df["feature"].apply(
    lambda x: extract_original_feature(x, num_cols, cat_cols)
)

coef_df["base_feature"] = coef_df["feature"].apply(
    lambda x: extract_original_feature(x, num_cols, cat_cols)
)


feature_importance_df = (
    selected_coef_df
    .groupby("base_feature", as_index=False)
    .agg(
        sum_abs_coef=("abs_coef", "sum"),
        max_abs_coef=("abs_coef", "max"),
        nonzero_count=("abs_coef", "size"),
    )
    .sort_values(["sum_abs_coef", "max_abs_coef"], ascending=False)
    .reset_index(drop=True)
)

selected_features_all = feature_importance_df["base_feature"].tolist()
selected_features_top = feature_importance_df.head(TOP_N)["base_feature"].tolist()

print("=" * 60)
print("TOP ORIGINAL FEATURES")
print(feature_importance_df.head(30))

print("=" * 60)
print("SELECTED ORIGINAL FEATURES (all nonzero):", len(selected_features_all))
print(f"SELECTED TOP {TOP_N} ORIGINAL FEATURES:", len(selected_features_top))


coef_df.to_csv(
    RESULT_OUTPUT_DIR / "best_features_logreg_filter.csv",
    index=False,
    encoding="utf-8-sig",
)

selected_coef_df.to_csv(
    TMP_OUTPUT_DIR / "selected_features_logreg_filter.csv",
    index=False,
    encoding="utf-8-sig",
)

feature_importance_df.to_csv(
    TMP_OUTPUT_DIR / "logreg_original_feature_importance.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame({"feature": selected_features_all}).to_csv(
    TMP_OUTPUT_DIR / "logreg_selected_features_all_nonzero.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame({"feature": selected_features_top}).to_csv(
    TMP_OUTPUT_DIR / "logreg_selected_features_top.csv",
    index=False,
    encoding="utf-8-sig",
)

with open(TMP_OUTPUT_DIR / "best_params_logreg_filter.json", "w", encoding="utf-8") as f:
    json.dump(study.best_params, f, ensure_ascii=False, indent=4)

n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
n_completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
n_convergence_trials = sum(
    bool(t.user_attrs.get("had_convergence_warning", False))
    for t in study.trials
    if t.state == optuna.trial.TrialState.COMPLETE
)

summary_df = pd.DataFrame([{
    "cv_best_roc_auc": study.best_value,
    "train_roc_auc": train_roc_auc,
    "test_roc_auc": test_roc_auc,
    "n_trials": len(study.trials),
    "n_pruned": n_pruned,
    "n_completed": n_completed,
    "n_convergence_trials": n_convergence_trials,
    "best_model_convergence_warning": best_model_had_convergence_warning,
    "n_selected_onehot_features": len(selected_coef_df),
    "n_selected_original_features": len(selected_features_all),
    "top_n_original_features": TOP_N,
}])

summary_df.to_csv(
    TMP_OUTPUT_DIR / "logreg_filter_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("=" * 60)
print("FILES SAVED:")