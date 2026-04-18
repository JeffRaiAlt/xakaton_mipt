from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (average_precision_score, precision_score, recall_score,
                             roc_auc_score, f1_score)


@dataclass
class QuickCatBoostModelCheckConfig:
    data_path: str
    output_path: str
    target: str = "buyout_flag"
    time_col: str = "sale_ts"
    random_state: int = 42
    test_size: float = 0.2
    n_splits: int = 5
    n_trials: int = 30
    thresholds: np.ndarray | None = None
    min_recall_0: float = 0.90
    min_precision_0: float = 0.20
    min_precision_1: float = 0.90

    def __post_init__(self) -> None:
        if self.thresholds is None:
            self.thresholds = np.arange(0.05, 0.96, 0.01)




class QuickCatBoostModelCheck:
    def __init__(self, config: QuickCatBoostModelCheckConfig) -> None:
        self.config = config
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []



    def do_work_cat(self) -> dict[str, Any]:
        df = self._load_data()
        train_df, test_df, features = self._split_data(df)
        self._init_feature_types(train_df[features])

        folds = self._generate_time_folds(train_df)
        best_params = self._tune_params(train_df, features, folds)
        best_thr = self._select_threshold(train_df, features, folds, best_params)

        X_train = train_df[features].copy()
        y_train = train_df[self.config.target].copy()
        X_test = test_df[features].copy()
        y_test = test_df[self.config.target].copy()

        X_train = self._prepare_cat_features(X_train)
        X_test = self._prepare_cat_features(X_test)

        model = CatBoostClassifier(**best_params)
        model.fit(
            X_train,
            y_train,
            cat_features=self.cat_cols,
            verbose=False,
        )

        p1_train = model.predict_proba(X_train)[:, 1]
        p0_train = 1.0 - p1_train
        p1_test = model.predict_proba(X_test)[:, 1]
        p0_test = 1.0 - p1_test

        self._save_class0_error_analysis(
            X_raw=X_train,
            y_true=y_train,
            p0=p0_train,
            p1=p1_train,
            threshold=best_thr,
            split_name="train",
        )

        self._save_class0_error_analysis(
            X_raw=X_test,
            y_true=y_test,
            p0=p0_test,
            p1=p1_test,
            threshold=best_thr,
            split_name="test",
        )

        feature_importance = pd.DataFrame(
            {
                "feature": features,
                "importance": model.get_feature_importance(),
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        return {
            "best_params": best_params,
            "best_threshold": best_thr,
            "train_metrics": self._collect_metrics(y_train, p0_train, p1_train, best_thr),
            "test_metrics": self._collect_metrics(y_test, p0_test, p1_test, best_thr),
            "feature_importance": feature_importance,
        }

    drop_cols = [
        "sale_date", "handed_to_delivery_ts",
    "issued_or_pvz_ts", "received_ts", "rejected_ts", "returned_ts",
        "days_handed_to_issued_pvz", "days_to_outcome", "closed_ts", "lead_closed_at",
        "contact_LTV", "lead_Квалификация лида", "lead_Дата возврата посылки на склад",
        "lead_Дата создания сделки", "lead_Дата перехода в Сборку",
        "lead_Дата перехода Передан в доставку",  "row_id",
        "lead_Квалификация лида", "lead_Дата получения денег на Р/С",
        "lead_Условный отказ", "days_sale_to_handed", "contact_Число сделок",
        "lead_Трек-номер СДЭК",
        "lead_Номер отправления СДЭК",
        "lead_Трек-номер",
        "contact_Трекинг",
        "lead_будущие покупки",
        "contact_id",
        "lead_clientID",
        "contact_responsible_user_id",
        "lead_responsible_user_id",
        "lead_URL страницы",
        "lead_ROISTAT_URL",
        "lead_ROISTAT_REFERRER",
        "lead_ROISTAT_FIELDS_ROISTAT",
        "lead_REFERER",
        "lead_ROISTAT_POS"
    ]

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path, low_memory=False)
        #df = df.drop(columns=self.drop_cols)
        df = df[df[self.config.target].notna()].copy()
        df[self.config.target] = df[self.config.target].astype(int)
        df[self.config.time_col] = pd.to_datetime(df[self.config.time_col], errors="coerce")
        df = df.dropna(subset=[self.config.time_col]).copy()
        # Это критичное место!!!!! Без него будут утечки
        df = df.sort_values(self.config.time_col).reset_index(drop=True)
        return df

    def _split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        split_idx = int(len(df) * (1.0 - self.config.test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        features = [
            c for c in df.columns
            if c not in [self.config.target, self.config.time_col, "row_id"]
        ]
        return train_df, test_df, features

    def _init_feature_types(self, X: pd.DataFrame) -> None:
        self.cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]

    def _prepare_cat_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.cat_cols:
            X[col] = X[col].astype("object").where(pd.notna(X[col]), "missing").astype(str)

        if self.num_cols:
            medians = X[self.num_cols].median()
            X[self.num_cols] = X[self.num_cols].fillna(medians)

        return X

    def _generate_time_folds(self, train_df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        indices = np.arange(len(train_df))
        folds = []

        for tr_idx, val_idx in tscv.split(indices):
            folds.append((tr_idx, val_idx))

        return folds

    def _objective(
        self,
        trial: optuna.Trial,
        train_df: pd.DataFrame,
        features: list[str],
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        w0 = trial.suggest_categorical("class_weight_0", [1.0, 1.25, 1.5, 2.0, 3.0])

        """params = {
            "iterations": trial.suggest_int("iterations", 150, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 10.0, log=True),
            "class_weights": [w0, 1.0],
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": self.config.random_state,
            "verbose": False,
        }"""

        params = {
            "iterations": trial.suggest_int("iterations", 200, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.02,
                                                 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0,
                                               log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0,
                                                   5.0),
            "border_count": trial.suggest_categorical("border_count",
                                                      [64, 128, 254]),
            "class_weights": [w0, 1.0],
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": self.config.random_state,
            "verbose": False,
        }

        scores: list[float] = []
        for tr_idx, val_idx in folds:
            X_tr = train_df.iloc[tr_idx][features].copy()
            y_tr = train_df.iloc[tr_idx][self.config.target].copy()
            X_val = train_df.iloc[val_idx][features].copy()
            y_val = train_df.iloc[val_idx][self.config.target].copy()

            X_tr = self._prepare_cat_features(X_tr)
            X_val = self._prepare_cat_features(X_val)

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                cat_features=self.cat_cols,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=50,
                verbose=False,
            )

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

        w0 = float(best_params.pop("class_weight_0"))
        best_params["class_weights"] = [w0, 1.0]

        best_params.update(
            {
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "random_seed": self.config.random_state,
                "verbose": False,
            }
        )
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
            X_tr = train_df.iloc[tr_idx][features].copy()
            y_tr = train_df.iloc[tr_idx][self.config.target].copy()
            X_val = train_df.iloc[val_idx][features].copy()
            y_val = train_df.iloc[val_idx][self.config.target].copy()

            X_tr = self._prepare_cat_features(X_tr)
            X_val = self._prepare_cat_features(X_val)

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                cat_features=self.cat_cols,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=50,
                verbose=False,
            )

            oof_p0.iloc[val_idx] = 1.0 - model.predict_proba(X_val)[:, 1]

        valid_mask = oof_p0.notna()
        y = train_df.loc[valid_mask, self.config.target].reset_index(drop=True)
        p0 = oof_p0.loc[valid_mask].reset_index(drop=True)

        best_thr = float(self.config.thresholds[0])
        best_score = -np.inf

        for thr in self.config.thresholds:
            pred = np.where(p0 >= thr, 0, 1)
            score = f1_score(y, pred, pos_label=0, zero_division=0)

            if score > best_score:
                best_score = score
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

    def _save_class0_error_analysis(
            self,
            X_raw: pd.DataFrame,
            y_true: pd.Series,
            p0: np.ndarray,
            p1: np.ndarray,
            threshold: float,
            split_name: str,
    ) -> None:
        output_dir = Path(self.config.output_path) / "error_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        y_true = y_true.reset_index(drop=True)
        X_raw = X_raw.reset_index(drop=True)

        pred_label = np.where(p0 >= threshold, 0, 1)

        error_df = X_raw.copy()
        error_df["y_true"] = y_true
        error_df["p0"] = p0
        error_df["p1"] = p1
        error_df["pred_label"] = pred_label

        error_df["error_type"] = np.select(
            [
                (error_df["y_true"] == 0) & (error_df["pred_label"] == 0),
                (error_df["y_true"] == 0) & (error_df["pred_label"] == 1),
                (error_df["y_true"] == 1) & (error_df["pred_label"] == 0),
                (error_df["y_true"] == 1) & (error_df["pred_label"] == 1),
            ],
            ["tp_0", "fn_0", "fp_0", "tn_0"],
            default="unknown",
        )

        error_df.to_csv(output_dir / f"{split_name}_full_error_analysis.csv", index=False)

        tp_0_df = error_df[error_df["error_type"] == "tp_0"].copy()
        fn_0_df = error_df[error_df["error_type"] == "fn_0"].copy()
        fp_0_df = error_df[error_df["error_type"] == "fp_0"].copy()
        tn_0_df = error_df[error_df["error_type"] == "tn_0"].copy()

        tp_0_df.to_csv(output_dir / f"{split_name}_tp_0.csv", index=False)
        fn_0_df.to_csv(output_dir / f"{split_name}_fn_0.csv", index=False)
        fp_0_df.to_csv(output_dir / f"{split_name}_fp_0.csv", index=False)
        tn_0_df.to_csv(output_dir / f"{split_name}_tn_0.csv", index=False)

        # =========================
        # CLUSTERING FP0
        # =========================
        if len(fp_0_df) >= 2:
            try:
                feature_cols = [c for c in self.num_cols + self.cat_cols if c in fp_0_df.columns]
                X_fp0 = fp_0_df[feature_cols].copy()

                # простая локальная подготовка:
                # numeric -> как есть
                # categorical -> category codes
                cluster_parts = []

                numeric_cols = [c for c in self.num_cols if c in X_fp0.columns]
                if numeric_cols:
                    X_num = X_fp0[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                    cluster_parts.append(X_num)

                categorical_cols = [c for c in self.cat_cols if c in X_fp0.columns]
                if categorical_cols:
                    X_cat = X_fp0[categorical_cols].copy()
                    for col in categorical_cols:
                        X_cat[col] = (
                            X_cat[col]
                            .astype("object")
                            .fillna("missing")
                            .astype("category")
                            .cat.codes
                        )
                    X_cat = X_cat.astype(float)
                    cluster_parts.append(X_cat)

                if cluster_parts:
                    X_cluster = pd.concat(cluster_parts, axis=1)

                    if X_cluster.shape[1] > 0:
                        X_scaled = StandardScaler().fit_transform(X_cluster)

                        n_components = min(10, X_scaled.shape[0], X_scaled.shape[1])
                        if n_components >= 1:
                            X_reduced = PCA(
                                n_components=n_components,
                                random_state=self.config.random_state,
                            ).fit_transform(X_scaled)

                            n_clusters = min(4, len(fp_0_df))
                            if n_clusters >= 2:
                                clusters = KMeans(
                                    n_clusters=n_clusters,
                                    random_state=self.config.random_state,
                                    n_init=10,
                                ).fit_predict(X_reduced)

                                fp_0_df["cluster"] = clusters
                                fp_0_df.to_csv(
                                    output_dir / f"{split_name}_fp0_with_clusters.csv",
                                    index=False,
                                )

                                summary_rows = []
                                for cluster_id in sorted(fp_0_df["cluster"].unique()):
                                    cluster_df = fp_0_df[fp_0_df["cluster"] == cluster_id]

                                    row = {
                                        "cluster": int(cluster_id),
                                        "size": int(len(cluster_df)),
                                        "share": float(len(cluster_df) / len(fp_0_df)),
                                        "mean_p0": float(cluster_df["p0"].mean()),
                                        "mean_p1": float(cluster_df["p1"].mean()),
                                    }

                                    for col in categorical_cols[:10]:
                                        vc = cluster_df[col].astype(str).value_counts(dropna=False)
                                        if not vc.empty:
                                            row[f"{col}_top"] = vc.index[0]
                                            row[f"{col}_top_share"] = float(vc.iloc[0] / len(cluster_df))

                                    summary_rows.append(row)

                                pd.DataFrame(summary_rows).to_csv(
                                    output_dir / f"{split_name}_fp0_cluster_summary.csv",
                                    index=False,
                                )

            except Exception as e:
                pd.DataFrame(
                    [{"stage": "fp0_clustering", "error": str(e)}]
                ).to_csv(output_dir / f"{split_name}_fp0_clustering_error.csv", index=False)

        numeric_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()

        if numeric_cols and not fn_0_df.empty and not tp_0_df.empty:
            tp_means = tp_0_df[numeric_cols].mean()
            fn_means = fn_0_df[numeric_cols].mean()

            pd.DataFrame(
                {
                    "feature": numeric_cols,
                    "tp0_mean": tp_means.values,
                    "fn0_mean": fn_means.values,
                    "signed_diff": (tp_means - fn_means).values,
                    "abs_diff": (tp_means - fn_means).abs().values,
                }
            ).sort_values("abs_diff", ascending=False).to_csv(
                output_dir / f"{split_name}_tp0_vs_fn0_numeric_diff.csv",
                index=False,
            )

        if numeric_cols and not fp_0_df.empty and not tn_0_df.empty:
            fp_means = fp_0_df[numeric_cols].mean()
            tn_means = tn_0_df[numeric_cols].mean()

            pd.DataFrame(
                {
                    "feature": numeric_cols,
                    "fp0_mean": fp_means.values,
                    "tn0_mean": tn_means.values,
                    "signed_diff": (fp_means - tn_means).values,
                    "abs_diff": (fp_means - tn_means).abs().values,
                }
            ).sort_values("abs_diff", ascending=False).to_csv(
                output_dir / f"{split_name}_fp0_vs_tn0_numeric_diff.csv",
                index=False,
            )

        cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
        rows = []

        if not fp_0_df.empty and not tn_0_df.empty:
            for col in cat_cols:
                fp_share = fp_0_df[col].astype(str).value_counts(normalize=True)
                tn_share = tn_0_df[col].astype(str).value_counts(normalize=True)
                all_vals = sorted(set(fp_share.index) | set(tn_share.index))

                for val in all_vals:
                    fp_val = float(fp_share.get(val, 0.0))
                    tn_val = float(tn_share.get(val, 0.0))
                    rows.append(
                        {
                            "feature": col,
                            "value": val,
                            "fp0_share": fp_val,
                            "tn0_share": tn_val,
                            "abs_diff": abs(fp_val - tn_val),
                            "signed_diff": fp_val - tn_val,
                        }
                    )

        if rows:
            pd.DataFrame(rows).sort_values("abs_diff", ascending=False).to_csv(
                output_dir / f"{split_name}_fp0_vs_tn0_categorical_diff.csv",
                index=False,
            )

        pd.DataFrame(
            [
                {"metric": "threshold", "value": threshold},
                {"metric": "total_rows", "value": len(error_df)},
                {"metric": "tp_0_count", "value": len(tp_0_df)},
                {"metric": "fn_0_count", "value": len(fn_0_df)},
                {"metric": "fp_0_count", "value": len(fp_0_df)},
                {"metric": "tn_0_count", "value": len(tn_0_df)},
            ]
        ).to_csv(output_dir / f"{split_name}_summary.csv", index=False)


def do_work_cat(
    data_path: str,
    output_path: str,
    target: str = "buyout_flag",
    time_col: str = "sale_ts",
    random_state: int = 42,
    test_size: float = 0.2,
    n_splits: int = 5,
    n_trials: int = 30,
    thresholds: np.ndarray | None = None,
    min_recall_0: float = 0.90,
    min_precision_0: float = 0.15,
    min_precision_1: float = 0.90,
) -> dict[str, Any]:
    config = QuickCatBoostModelCheckConfig(
        data_path=data_path,
        output_path=output_path,
        target=target,
        time_col=time_col,
        random_state=random_state,
        test_size=test_size,
        n_splits=n_splits,
        n_trials=n_trials,
        thresholds=thresholds,
        min_recall_0=min_recall_0,
        min_precision_0=min_precision_0,
        min_precision_1=min_precision_1,
    )
    return QuickCatBoostModelCheck(config).do_work_cat()