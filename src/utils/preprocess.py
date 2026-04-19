import numpy as np
import pandas as pd
import re
from catboost import CatBoostClassifier, Pool, EFstrType
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from src.utils.contact_code_utils import (
    exctract_code_pvz,
    expand_cities_by_comma,
    get_region,
)


def preprocess_initial_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["buyout_flag"].isna()].copy()
    df["buyout_flag"] = df["buyout_flag"].astype(int)
    df["contact_pvz_code"] = df["contact_pvz_code"].replace("HOME", "unknown")
    df["contact_has_pvz_code"] = np.where(
        df["contact_pvz_code"] == "unknown", 0, 1
    )
    col = "contact_pvz_code"
    df[col] = df[col].astype(str)
    top15 = df.loc[df[col] != "unknown", col].value_counts().head(15).index
    df[col] = df[col].apply(
        lambda x: x if x == "unknown" else (x if x in top15 else "rare_code")
    )
    df = df.drop(
        columns=[
            "contact_loyalty",
            "row_id",
            "contact_LTV",
            "has_contact_LTV",
            "buyout_flag_lag30",
            "buyout_flag_lag60",
            "buyout_flag_ma30",
            "lead_utm_content",
            "sale_date",
            "lead_pipeline_id",
        ]
    )
    #df[["sale_ts", "lead_created_ts"]] = df[
    #    ["sale_ts", "lead_created_ts"]
    #].astype("datetime64[ns]")
    df[["sale_ts"]] = df[
        ["sale_ts"]
     ].astype("datetime64[ns]")
    return df


def split_dataset(df: pd.DataFrame):
    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train.drop(columns=["buyout_flag", "sale_ts"])
    y_train = train["buyout_flag"]

    X_test = test.drop(columns=["buyout_flag", "sale_ts"])
    y_test = test["buyout_flag"]
    return X_train, y_train, X_test, y_test


def split_dataset_with_val(df: pd.DataFrame, val_size=0.1):
    df = df.sort_values(by="sale_ts").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(train_end * (1 - val_size))

    train = df.iloc[:val_end]
    val = df.iloc[val_end:train_end]
    test = df.iloc[train_end:]

    X_train = train.drop(columns=["buyout_flag", "sale_ts"])
    y_train = train["buyout_flag"]

    X_val = val.drop(columns=["buyout_flag", "sale_ts"])
    y_val = val["buyout_flag"]

    X_test = test.drop(columns=["buyout_flag", "sale_ts"])
    y_test = test["buyout_flag"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def remove_log_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lead_Стоимость доставки"] = (
        df["lead_Стоимость доставки"]
        .replace("-", np.nan)
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df["lead_Стоимость доставки"] = pd.to_numeric(
        df["lead_Стоимость доставки"], errors="coerce"
    )
    df[["lead_Стоимость доставки", "lead_Масса (гр)", "lead_Высота"]] = df[
        ["lead_Стоимость доставки", "lead_Масса (гр)", "lead_Высота"]
    ].fillna(-1)
    return df.drop(
        columns=["delivery_cost_log", "lead_height_log", "lead_mass_log"]
    )


def evaluate(model, X, y):
    y_refuse = (y == 0).astype(int)
    proba_refuse = model.predict_proba(X)[:, 1]
    return {
        "ROC_AUC": roc_auc_score(y_refuse, proba_refuse),
        "PR_AUC": average_precision_score(y_refuse, proba_refuse),
        "LogLoss": log_loss(y_refuse, model.predict_proba(X)[:, 1]),
    }


def train_catboost_model(X_train, y_train, X_val, y_val, X_test, y_test):
    # категориальные признаки
    cat_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    #  Переворачиваем метки: теперь positive класс = отказ от выкупа (buyout_flag == 0)
    y_train_refuse = (y_train == 0).astype(int)
    y_val_refuse = (y_val == 0).astype(int)

    # модель
    model = CatBoostClassifier(
        iterations=2000,
        depth=6,
        learning_rate=0.1,
        verbose=100,
        cat_features=cat_features,
        random_state=42,
        class_weights={0: 1, 1: 5},
        eval_metric="PRAUC",
        early_stopping_rounds=100,
    )

    # обучение
    model.fit(X_train, y_train_refuse, eval_set=(X_val, y_val_refuse), use_best_model=True)

    # метрики
    metrics = evaluate(model, X_test, y_test)

    return model, metrics


def get_feature_importance_df(
    model, X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    cat_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)

    importances = model.get_feature_importance(
        data=train_pool, type=EFstrType.FeatureImportance
    )

    fi = pd.DataFrame(
        {"feature": X_train.columns, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    return fi


def transform_contact_region_pvz(
    df: pd.DataFrame, pvz_data: pd.DataFrame
) -> pd.DataFrame:
    # --- 1. извлекаем код ПВЗ ---
    df["contact_Код ПВЗ"] = [
        exctract_code_pvz(x) for x in df["contact_Код ПВЗ"]
    ]

    # --- 2. расширяем города ---
    pvz_data = expand_cities_by_comma(pvz_data)

    # --- 3. строим словарь prefix -> region ---
    pvz_dict = {}
    for pvz_code, region in zip(pvz_data.iloc[:, 2], pvz_data.iloc[:, 0]):
        match = re.match(r"^([A-Z]+)", str(pvz_code))
        if match:
            pvz_dict[match.group(1)] = region

    # --- 4. получаем регион ---
    df["contact_region_pvz"] = df["contact_Код ПВЗ"].apply(
        lambda x: get_region(x, pvz_dict)
    )

    # --- 5. оставляем top-15 + unknown ---
    col = "contact_region_pvz"

    mask = df[col] != "unknown"
    top15 = df.loc[mask, col].value_counts().head(15).index

    df[col] = df[col].where(
        df[col].isin(top15) | (df[col] == "unknown"), "rare_region"
    )

    return df
