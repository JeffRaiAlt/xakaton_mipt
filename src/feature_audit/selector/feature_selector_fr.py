from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class FeatureSelectorConfig:
    data_path: str
    output_dir: str
    target: str = "buyout_flag"
    random_state: int = 42
    test_size: float = 0.2
    n_splits: int = 5
    n_trials: int = 25
    top_n: int = 50
    start_date: str | None = None
    end_date: str | None = None
    date_filter_col: str = "lead_created_dt"
    drop_cols: tuple[str, ...] | None = None,


class RandomForestFeatureSelector:
    def __init__(self, config: FeatureSelectorConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path, low_memory=False)
        df = df[df[self.config.target].notna()].copy()
        df[self.config.target] = df[self.config.target].astype(int)

        date_col = self.config.date_filter_col
        if date_col in df.columns and self.config.start_date and self.config.end_date:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            try:
                df[date_col] = df[date_col].dt.tz_localize(None)
            except TypeError:
                pass
            df = df[df[date_col].between(self.config.start_date, self.config.end_date)].copy()

        df = df.drop(columns=list(self.config.drop_cols), errors="ignore")
        return df

    @staticmethod
    def _encode(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = train_df.copy()
        test_df = test_df.copy()

        cat_cols = train_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        for col in cat_cols:
            tr = train_df[col].astype(str)
            te = test_df[col].astype(str)
            freq = tr.value_counts(normalize=True)
            train_df[col] = tr.map(freq).fillna(0.0).astype(float)
            test_df[col] = te.map(freq).fillna(0.0).astype(float)

        return train_df, test_df

    def _objective(self, trial: optuna.Trial, x: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 12),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": self.config.random_state,
            "n_jobs": -1,
        }

        scores: list[float] = []
        for train_idx, val_idx in cv.split(x, y):
            x_tr = x.iloc[train_idx]
            x_val = x.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = RandomForestClassifier(**params)
            model.fit(x_tr, y_tr)
            p1 = model.predict_proba(x_val)[:, 1]
            scores.append(roc_auc_score(y_val, p1))

        return float(sum(scores) / len(scores))

    def prepare_features_fr(self) -> dict:
        df = self._load_data()
        x = df.drop(columns=[self.config.target]).copy()
        y = df[self.config.target].copy()

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        x_train_enc, x_test_enc = self._encode(x_train, x_test)

        cv = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, x_train_enc, y_train, cv),
            n_trials=self.config.n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params.copy()
        best_params["random_state"] = self.config.random_state
        best_params["n_jobs"] = -1

        model = RandomForestClassifier(**best_params)
        model.fit(x_train_enc, y_train)

        train_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train_enc)[:, 1])
        test_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test_enc)[:, 1])

        feature_importance_df = pd.DataFrame(
            {
                "feature": x_train_enc.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        selected_features = feature_importance_df.head(self.config.top_n)["feature"].tolist()

        feature_importance_df.to_csv(
            self.output_dir / "best_features_rf_filter.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame({"feature": selected_features}).to_csv(
            self.output_dir / "rf_selected_features_top.csv",
            index=False,
            encoding="utf-8-sig",
        )
        with open(self.output_dir / "best_params_rf_filter.json", "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, ensure_ascii=False, indent=4)

        return {
            "best_params": best_params,
            "train_roc_auc": float(train_roc_auc),
            "test_roc_auc": float(test_roc_auc),
            "selected_features": selected_features,
            "feature_importance": feature_importance_df,
            "output_dir": str(self.output_dir),
        }


def prepare_features_fr(
    data_path: str,
    output_dir: str,
    target: str = "buyout_flag",
    random_state: int = 42,
    test_size: float = 0.2,
    n_splits: int = 5,
    n_trials: int = 25,
    top_n: int = 50,
    start_date: str | None = None,
    end_date: str | None = None,
    date_filter_col: str = "lead_created_dt",
    drop_cols: tuple[str, ...] | None = None,
) -> dict:
    config = FeatureSelectorConfig(
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
    return RandomForestFeatureSelector(config).prepare_features_fr()
