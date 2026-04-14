from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


@dataclass
class QuickModelCheckConfig:
    data_path: str
    target: str = "buyout_flag"
    time_col: str = "sale_ts"
    random_state: int = 42
    test_size: float = 0.2
    n_splits: int = 5
    n_trials: int = 30
    thresholds: np.ndarray | None = None
    min_recall_0: float = 0.90
    min_precision_1: float = 0.90

    def __post_init__(self) -> None:
        if self.thresholds is None:
            self.thresholds = np.arange(0.05, 0.96, 0.01)


class QuickRandomForestModelCheck:
    def __init__(self, config: QuickModelCheckConfig) -> None:
        self.config = config
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []

    def do_work(self) -> dict[str, Any]:
        df = self._load_data()
        train_df, test_df, features = self._split_data(df)
        self._init_feature_types(train_df[features])

        folds = self._generate_time_folds(train_df)
        best_params = self._tune_params(train_df, features, folds)
        best_thr = self._select_threshold(train_df, features, folds, best_params)

        X_train_raw = train_df[features].copy()
        y_train = train_df[self.config.target].copy()
        X_test_raw = test_df[features].copy()
        y_test = test_df[self.config.target].copy()

        X_train, medians, freq_maps = self._preprocess_fit_transform(X_train_raw)
        X_test = self._preprocess_transform(X_test_raw, medians, freq_maps)

        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)

        p1_train = model.predict_proba(X_train)[:, 1]
        p0_train = 1.0 - p1_train
        p1_test = model.predict_proba(X_test)[:, 1]
        p0_test = 1.0 - p1_test

        return {
            "best_params": best_params,
            "best_threshold": best_thr,
            "train_metrics": self._collect_metrics(y_train, p0_train, p1_train, best_thr),
            "test_metrics": self._collect_metrics(y_test, p0_test, p1_test, best_thr),
            "feature_importance": pd.DataFrame(
                {
                    "feature": features,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False).reset_index(drop=True),
        }

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path, low_memory=False)
        df = df[df[self.config.target].notna()].copy()
        df[self.config.target] = df[self.config.target].astype(int)
        df[self.config.time_col] = pd.to_datetime(df[self.config.time_col], errors="coerce")
        df = df.dropna(subset=[self.config.time_col]).copy()
        df = df.sort_values(self.config.time_col).reset_index(drop=True)
        return df

    def _split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        split_idx = int(len(df) * (1.0 - self.config.test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        features = [c for c in df.columns if c not in [self.config.target, self.config.time_col]]
        return train_df, test_df, features

    def _init_feature_types(self, X: pd.DataFrame) -> None:
        self.cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]

    def _preprocess_fit_transform(
        self,
        X: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, dict[str, pd.Series]]:
        X = X.copy()

        for col in self.cat_cols:
            X[col] = X[col].astype("object").where(pd.notna(X[col]), "missing").astype(str)

        medians = X[self.num_cols].median() if self.num_cols else pd.Series(dtype=float)
        if self.num_cols:
            X[self.num_cols] = X[self.num_cols].fillna(medians)

        freq_maps: dict[str, pd.Series] = {}
        for col in self.cat_cols:
            freq = X[col].value_counts(normalize=True)
            freq_maps[col] = freq
            X[col] = X[col].map(freq).fillna(0.0)

        return X, medians, freq_maps

    def _preprocess_transform(
        self,
        X: pd.DataFrame,
        medians: pd.Series,
        freq_maps: dict[str, pd.Series],
    ) -> pd.DataFrame:
        X = X.copy()

        for col in self.cat_cols:
            X[col] = X[col].astype("object").where(pd.notna(X[col]), "missing").astype(str)

        if self.num_cols:
            X[self.num_cols] = X[self.num_cols].fillna(medians)

        for col in self.cat_cols:
            X[col] = X[col].map(freq_maps[col]).fillna(0.0)

        return X

    def _generate_time_folds(self, train_df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(train_df))
        fold_size = len(train_df) // self.config.n_splits
        folds: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(self.config.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.config.n_splits - 1 else len(train_df)
            train_idx = indices[:start]
            val_idx = indices[start:end]
            if len(train_idx) > 0 and len(val_idx) > 0:
                folds.append((train_idx, val_idx))

        return folds

    def _objective(
        self,
        trial: optuna.Trial,
        train_df: pd.DataFrame,
        features: list[str],
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
            "random_state": self.config.random_state,
            "n_jobs": -1,
        }

        scores: list[float] = []
        for tr_idx, val_idx in folds:
            X_tr_raw = train_df.iloc[tr_idx][features].copy()
            y_tr = train_df.iloc[tr_idx][self.config.target].copy()
            X_val_raw = train_df.iloc[val_idx][features].copy()
            y_val = train_df.iloc[val_idx][self.config.target].copy()

            X_tr, medians, freq_maps = self._preprocess_fit_transform(X_tr_raw)
            X_val = self._preprocess_transform(X_val_raw, medians, freq_maps)

            model = RandomForestClassifier(**params)
            model.fit(X_tr, y_tr)

            p0_val = 1.0 - model.predict_proba(X_val)[:, 1]
            y0_val = (y_val.reset_index(drop=True) == 0).astype(int)
            scores.append(float(average_precision_score(y0_val, p0_val)))

        return float(np.mean(scores))

    def _tune_params(
        self,
        train_df: pd.DataFrame,
        features: list[str],
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: self._objective(trial, train_df, features, folds),
            n_trials=self.config.n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params.copy()
        best_params["random_state"] = self.config.random_state
        best_params["n_jobs"] = -1
        return best_params

    def _select_threshold(
        self,
        train_df: pd.DataFrame,
        features: list[str],
        folds: list[tuple[np.ndarray, np.ndarray]],
        params: dict[str, Any],
    ) -> float:
        oof_p0 = pd.Series(index=train_df.index, dtype=float)

        for tr_idx, val_idx in folds:
            X_tr_raw = train_df.iloc[tr_idx][features].copy()
            y_tr = train_df.iloc[tr_idx][self.config.target].copy()
            X_val_raw = train_df.iloc[val_idx][features].copy()

            X_tr, medians, freq_maps = self._preprocess_fit_transform(X_tr_raw)
            X_val = self._preprocess_transform(X_val_raw, medians, freq_maps)

            model = RandomForestClassifier(**params)
            model.fit(X_tr, y_tr)
            oof_p0.iloc[val_idx] = 1.0 - model.predict_proba(X_val)[:, 1]

        valid_mask = oof_p0.notna()
        y = train_df.loc[valid_mask, self.config.target].reset_index(drop=True)
        p0 = oof_p0.loc[valid_mask].reset_index(drop=True)

        best_thr = float(self.config.thresholds[0])
        best_score = -np.inf

        for thr in self.config.thresholds:
            pred = np.where(p0 >= thr, 0, 1)
            recall_0 = recall_score(y, pred, pos_label=0, zero_division=0)
            precision_1 = precision_score(y, pred, pos_label=1, zero_division=0)
            precision_0 = precision_score(y, pred, pos_label=0, zero_division=0)

            if recall_0 >= self.config.min_recall_0 and precision_1 >= self.config.min_precision_1:
                score = precision_0
                if score > best_score:
                    best_score = score
                    best_thr = float(thr)

        if best_score == -np.inf:
            for thr in self.config.thresholds:
                pred = np.where(p0 >= thr, 0, 1)
                precision_0 = precision_score(y, pred, pos_label=0, zero_division=0)
                if precision_0 > best_score:
                    best_score = precision_0
                    best_thr = float(thr)

        return best_thr

    def _collect_metrics(
        self,
        y: pd.Series,
        p0: np.ndarray,
        p1: np.ndarray,
        thr: float,
    ) -> dict[str, float]:
        y = y.reset_index(drop=True)
        pred = np.where(p0 >= thr, 0, 1)
        y0 = (y == 0).astype(int)

        return {
            "pr_auc_0": float(average_precision_score(y0, p0)),
            "roc_auc": float(roc_auc_score(y, p1)),
            "recall_0": float(recall_score(y, pred, pos_label=0, zero_division=0)),
            "precision_0": float(precision_score(y, pred, pos_label=0, zero_division=0)),
            "recall_1": float(recall_score(y, pred, pos_label=1, zero_division=0)),
            "precision_1": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        }


def do_work(
    data_path: str,
    target: str = "buyout_flag",
    time_col: str = "sale_ts",
    random_state: int = 42,
    test_size: float = 0.2,
    n_splits: int = 5,
    n_trials: int = 30,
    thresholds: np.ndarray | None = None,
    min_recall_0: float = 0.90,
    min_precision_1: float = 0.90,
) -> dict[str, Any]:
    config = QuickModelCheckConfig(
        data_path=data_path,
        target=target,
        time_col=time_col,
        random_state=random_state,
        test_size=test_size,
        n_splits=n_splits,
        n_trials=n_trials,
        thresholds=thresholds,
        min_recall_0=min_recall_0,
        min_precision_1=min_precision_1,
    )
    return QuickRandomForestModelCheck(config).do_work()
