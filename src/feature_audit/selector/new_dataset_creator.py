import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("../../").resolve()
sys.path.append(str(PROJECT_ROOT))


LOGREG_PATH = "result/best_features_logreg_filter.csv"
RF_PATH = "result/best_features_rf_filter.csv"
CATBOOST_PATH = "result/best_features_catboost.csv"

RESULT_OUTPUT_DIR = Path("out")
RESULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ALL_SCORES = RESULT_OUTPUT_DIR / "final_feature_scores.csv"
OUTPUT_SELECTED = RESULT_OUTPUT_DIR / "final_selected_features.csv"

size = 60

TOP_K_LOGREG = size
TOP_K_RF = size
TOP_K_CB = size

FINAL_TOP_N = 40
MIN_VOTES = 2

W_LOGREG = 40
W_RF = 10
W_CB = 50



def prepare_rank_scores(df, feature_col, score_col, top_k, prefix):
    """
    Оставляет top_k признаков, присваивает rank и нормированный rank-score.
    """
    temp = df[[feature_col, score_col]].copy()
    temp = temp.dropna(subset=[feature_col])
    temp[feature_col] = temp[feature_col].astype(str)

    temp = temp.sort_values(score_col, ascending=False).reset_index(drop=True)
    temp = temp.drop_duplicates(subset=[feature_col], keep="first")
    temp = temp.head(top_k).reset_index(drop=True)

    temp[f"rank_{prefix}"] = np.arange(1, len(temp) + 1)
    temp[f"score_{prefix}"] = (top_k - temp[f"rank_{prefix}"] + 1) / top_k
    temp[f"vote_{prefix}"] = 1

    temp = temp.rename(columns={feature_col: "feature"})
    return temp[["feature", f"rank_{prefix}", f"score_{prefix}", f"vote_{prefix}"]]



df_lr_raw = pd.read_csv(LOGREG_PATH)
df_rf_raw = pd.read_csv(RF_PATH)
df_cb_raw = pd.read_csv(CATBOOST_PATH)

# LogReg: base_feature + sum_abs_coef
df_lr = prepare_rank_scores(
    df=df_lr_raw,
    feature_col="base_feature",
    score_col="abs_coef",
    top_k=TOP_K_LOGREG,
    prefix="lr"
)

# RF: feature + importance
df_rf = prepare_rank_scores(
    df=df_rf_raw,
    feature_col="feature",
    score_col="importance",
    top_k=TOP_K_RF,
    prefix="rf"
)

# CatBoost: feature + importance
df_cb = prepare_rank_scores(
    df=df_cb_raw,
    feature_col="feature",
    score_col="final_importance",
    top_k=TOP_K_CB,
    prefix="cb"
)


df_final = df_lr.merge(df_rf, on="feature", how="outer")
df_final = df_final.merge(df_cb, on="feature", how="outer")

for col in [
    "rank_lr", "score_lr", "vote_lr",
    "rank_rf", "score_rf", "vote_rf",
    "rank_cb", "score_cb", "vote_cb"
]:
    if col in df_final.columns:
        df_final[col] = df_final[col].fillna(0)

df_final["votes"] = (
    df_final["vote_lr"] +
    df_final["vote_rf"] +
    df_final["vote_cb"]
)

df_final["final_score"] = (
    W_LOGREG * df_final["score_lr"] +
    W_RF * df_final["score_rf"] +
    W_CB * df_final["score_cb"]
)

df_final = df_final.sort_values(
    by=["votes", "final_score"],
    ascending=[False, False]
).reset_index(drop=True)


# сначала устойчивые признаки
df_selected = df_final[df_final["votes"] >= MIN_VOTES].copy()

# если мало — добираем по final_score
if len(df_selected) < FINAL_TOP_N:
    already_selected = set(df_selected["feature"])
    df_extra = df_final[~df_final["feature"].isin(already_selected)].copy()
    n_to_add = FINAL_TOP_N - len(df_selected)
    df_extra = df_extra.head(n_to_add)
    df_selected = pd.concat([df_selected, df_extra], ignore_index=True)

# если много — режем
df_selected = df_selected.sort_values(
    by=["votes", "final_score"],
    ascending=[False, False]
).head(FINAL_TOP_N).reset_index(drop=True)


df_final.to_csv(OUTPUT_ALL_SCORES, index=False, encoding="utf-8-sig")
df_selected.to_csv(OUTPUT_SELECTED, index=False, encoding="utf-8-sig")

print("=" * 60)
print("DONE")
print(f"Total unique features in union: {len(df_final)}")
print(f"Selected final features: {len(df_selected)}")
print(f"Saved: {OUTPUT_ALL_SCORES}")
print(f"Saved: {OUTPUT_SELECTED}")

print("\nTop final features:")
print(df_selected.head(20))


# пути
DATA_PATH = "../../../assembled_outputs/final_dataset_from_notebooks.csv"
FEATURES_PATH = "./out/final_selected_features.csv"
OUTPUT_PATH = "./out/final_dataset_reduced.csv"

TARGET = "buyout_flag"

# загрузка
df = pd.read_csv(DATA_PATH)
selected_features = pd.read_csv(FEATURES_PATH)["feature"].tolist()

# todo сделать очистку в одном месте в самом начале
START_DATE = pd.Timestamp("2025-03-01")
END_DATE = pd.Timestamp("2026-03-29")

df["lead_created_dt"] = pd.to_datetime(df["lead_created_dt"], errors="coerce")
try:
    df["lead_created_dt"] = df["lead_created_dt"].dt.tz_localize(None)
except TypeError:
    pass

df = df[df["lead_created_dt"].between(START_DATE, END_DATE)].copy()

# важно: добавляем target вручную
final_cols = selected_features + ["sale_ts", TARGET]
final_cols = list(dict.fromkeys(final_cols))




df_final = df[final_cols].copy()
print("lead_group_quality" in df_final.columns)

#todo мам потенциально утечка, убрать
if "lead_group_quality" in df_final.columns:
    df_final = df_final.drop(columns=["lead_group_quality"])
print("lead_group_quality" in df_final.columns)
df_final.to_csv(OUTPUT_PATH, index=False)

print("Final dataset shape:", df_final.shape)