from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
import optuna

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score


@dataclass
class FeatureSelectorCatBoostConfig:
    data_path: str
    output_dir: str
    target: str = "buyout_flag"
    random_state: int = 42
    test_size: float = 0.2
    n_splits: int = 5
    n_trials: int = 10
    top_n: int = 50
    start_date: str | None = None
    end_date: str | None = None
    date_filter_col: str | None = "lead_created_dt"
    drop_cols: tuple[str, ...] | None = None



class FeatureSelectorCatBoost:
    def __init__(self, config: FeatureSelectorCatBoostConfig) -> None:
        self.config = config

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path, low_memory=False)
        df = df[df[self.config.target].notna()].copy()
        df[self.config.target] = df[self.config.target].astype(int)

        if self.config.date_filter_col and self.config.date_filter_col in df.columns:
            df[self.config.date_filter_col] = pd.to_datetime(
                df[self.config.date_filter_col],
                errors="coerce",
            )
            try:
                df[self.config.date_filter_col] = df[self.config.date_filter_col].dt.tz_localize(None)
            except TypeError:
                pass

            if self.config.start_date is not None:
                df = df[df[self.config.date_filter_col] >= pd.Timestamp(self.config.start_date)].copy()
            if self.config.end_date is not None:
                df = df[df[self.config.date_filter_col] <= pd.Timestamp(self.config.end_date)].copy()

        df = df.drop(columns=list(self.config.drop_cols), errors="ignore")
        return df

    @staticmethod
    def _preprocess_for_catboost(df: pd.DataFrame) -> pd.DataFrame:
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

    def _prepare_data(self, df: pd.DataFrame):
        X = df.drop(columns=[self.config.target]).copy()
        y = df[self.config.target].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        X_train = self._preprocess_for_catboost(X_train)
        X_test = self._preprocess_for_catboost(X_test)

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

        return X_train, X_test, y_train, y_test, cat_cols

    def _objective(self, trial, X_train, y_train, cat_cols, cv) -> float:
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
                ["Balanced", "SqrtBalanced"],
            ),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 0,
            "random_seed": self.config.random_state,
        }

        scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
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
            scores.append(roc_auc_score(y_val, p1_val))

        return float(sum(scores) / len(scores))

    def prepare_features_cat(self) -> dict:
        df = self._load_data()
        X_train, X_test, y_train, y_test, cat_cols = self._prepare_data(df)

        cv = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, cat_cols, cv),
            n_trials=self.config.n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params.copy()
        best_params.update(
            {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": 0,
                "random_seed": self.config.random_state,
            }
        )

        model = CatBoostClassifier(**best_params)
        model.fit(X_train, y_train, cat_features=cat_cols)

        train_pred_proba = model.predict_proba(X_train)[:, 1]
        test_pred_proba = model.predict_proba(X_test)[:, 1]

        train_roc_auc = roc_auc_score(y_train, train_pred_proba)
        test_roc_auc = roc_auc_score(y_test, test_pred_proba)

        feature_importance_df = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.get_feature_importance(),
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        selected_features = feature_importance_df.head(self.config.top_n)["feature"].tolist()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_importance_df.to_csv(
            output_dir / "best_features_catboost.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame({"feature": selected_features}).to_csv(
            output_dir / "catboost_selected_features_top.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with open(output_dir / "best_params_catboost.json", "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, ensure_ascii=False, indent=4)

        return {
            "best_params": study.best_params,
            "train_roc_auc": float(train_roc_auc),
            "test_roc_auc": float(test_roc_auc),
            "selected_features": selected_features,
            "feature_importance": feature_importance_df,
            "output_dir": str(output_dir),
        }


def prepare_features_cat(
    data_path: str,
    output_dir: str,
    target: str = "buyout_flag",
    random_state: int = 42,
    test_size: float = 0.2,
    n_splits: int = 5,
    n_trials: int = 10,
    top_n: int = 50,
    start_date: str | None = None,
    end_date: str | None = None,
    date_filter_col: str | None = "lead_created_dt",
    drop_cols: tuple[str, ...] | None = None,
) -> dict:
    config = FeatureSelectorCatBoostConfig(
        data_path=data_path,
        output_dir=output_dir,
        target=target,
        random_state=random_state,
        test_size=test_size,
        n_splits=n_splits,
        n_trials=n_trials,
        top_n=top_n,
        start_date=start_date,
        end_date=end_date,
        date_filter_col=date_filter_col,
        drop_cols=drop_cols,
    )
    return FeatureSelectorCatBoost(config).prepare_features_cat()
