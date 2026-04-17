from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class FeatureSelectorLogRegConfig:
    data_path: str
    output_dir: str
    target: str = "buyout_flag"
    random_state: int = 42
    test_size: float = 0.2
    n_splits: int = 3
    n_trials: int = 15
    top_n: int = 50
    start_date: str | None = None
    end_date: str | None = None
    date_filter_col: str | None = "lead_created_dt"
    drop_cols: tuple[str, ...] | None = None,


class FeatureSelectorLogReg:
    def __init__(self, config: FeatureSelectorLogRegConfig) -> None:
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

    def _prepare_data(self, df: pd.DataFrame):
        X = df.drop(columns=[self.config.target]).copy()
        y = df[self.config.target].copy()

        cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=["int64", "int32", "float64", "float32", "bool"]).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        preprocessor = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                ),
            ]
        )

        return X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols


    def _objective(self, trial, X_train, y_train, preprocessor) -> float:
        class_weight_option = trial.suggest_categorical(
            "class_weight",
            [
                "none",
                "balanced",
                "w2",
                "w3",
                "w5",
            ]
        )

        if class_weight_option == "none":
            class_weight = None
        elif class_weight_option == "balanced":
            class_weight = "balanced"
        elif class_weight_option == "w2":
            class_weight = {0: 2, 1: 1}
        elif class_weight_option == "w3":
            class_weight = {0: 3, 1: 1}
        elif class_weight_option == "w5":
            class_weight = {0: 5, 1: 1}

        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        solver="saga",
                        C=trial.suggest_float("C", 1e-2, 3.0, log=True),
                        l1_ratio=trial.suggest_categorical("l1_ratio", [0.5, 0.7, 0.9, 1.0]),
                        tol=trial.suggest_categorical("tol", [1e-3, 3e-3]),
                        max_iter=2000,
                        class_weight=class_weight,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

        cv = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            p1_val = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, p1_val))

        return float(sum(scores) / len(scores))

    @staticmethod
    def _extract_original_feature(feature_name: str, cat_cols: list[str]) -> str:
        if feature_name.startswith("num__"):
            return feature_name.replace("num__", "", 1)

        if feature_name.startswith("cat__"):
            base = feature_name.replace("cat__", "", 1)
            matches = [col for col in cat_cols if base == col or base.startswith(col + "_")]
            if matches:
                return max(matches, key=len)
            return base

        return feature_name

    def prepare_features_reg(self) -> dict:
        df = self._load_data()
        X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = self._prepare_data(df)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, preprocessor),
            n_trials=self.config.n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params.copy()

        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        solver="saga",
                        C=best_params["C"],
                        l1_ratio=best_params["l1_ratio"],
                        tol=best_params["tol"],
                        max_iter=5000,
                        class_weight=None,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        train_pred_proba = model.predict_proba(X_train)[:, 1]
        test_pred_proba = model.predict_proba(X_test)[:, 1]

        train_roc_auc = roc_auc_score(y_train, train_pred_proba)
        test_roc_auc = roc_auc_score(y_test, test_pred_proba)

        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        coefs = model.named_steps["classifier"].coef_[0]

        coef_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coef": coefs,
                "abs_coef": np.abs(coefs),
            }
        ).sort_values("abs_coef", ascending=False).reset_index(drop=True)

        selected_coef_df = coef_df[coef_df["coef"] != 0].copy().reset_index(drop=True)

        selected_coef_df["base_feature"] = selected_coef_df["feature"].apply(
            lambda x: self._extract_original_feature(x, cat_cols)
        )

        feature_importance_df = (
            selected_coef_df.groupby("base_feature", as_index=False)
            .agg(
                sum_abs_coef=("abs_coef", "sum"),
                max_abs_coef=("abs_coef", "max"),
                nonzero_count=("abs_coef", "size"),
            )
            .sort_values(["sum_abs_coef", "max_abs_coef"], ascending=False)
            .reset_index(drop=True)
        )

        selected_features = feature_importance_df.head(self.config.top_n)["base_feature"].tolist()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_importance_df.to_csv(
            output_dir / "best_features_logreg_filter.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame({"feature": selected_features}).to_csv(
            output_dir / "logreg_selected_features_top.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with open(output_dir / "best_params_logreg_filter.json", "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, ensure_ascii=False, indent=4)

        return {
            "best_params": study.best_params,
            "train_roc_auc": float(train_roc_auc),
            "test_roc_auc": float(test_roc_auc),
            "selected_features": selected_features,
            "feature_importance": feature_importance_df,
            "output_dir": str(output_dir),
        }


def prepare_features_reg(
    data_path: str,
    output_dir: str,
    target: str = "buyout_flag",
    random_state: int = 42,
    test_size: float = 0.2,
    n_splits: int = 3,
    n_trials: int = 15,
    top_n: int = 50,
    start_date: str | None = None,
    end_date: str | None = None,
    date_filter_col: str | None = "lead_created_dt",
    drop_cols: tuple[str, ...] | None = None,
) -> dict:
    config = FeatureSelectorLogRegConfig(
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
    return FeatureSelectorLogReg(config).prepare_features_reg()
