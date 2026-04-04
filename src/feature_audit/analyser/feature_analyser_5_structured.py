from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseAnalyzer, AnalyzerResult

class LeadFeatureEngineeringAnalyzer(BaseAnalyzer):
    """Структурированный feature engineering для набора lead_*.

    Особенности реализации:
    - логика разбита на небольшие методы;
    - удаления вынесены в конец;
    - есть проверки обязательных полей;
    - для дат действует допустимое окно [2025-03-01, 2026-03-29];
    - на выходе возвращаются подробные метаданные со статистикой.
    """

    name = "lead_feature_engineering"

    def __init__(
        self,
        start: str | pd.Timestamp = "2025-03-01",
        end: str | pd.Timestamp = "2026-03-29",
        target_col: str = "buyout_flag",
    ) -> None:
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.target_col = target_col

        self.required_columns = [
            "lead_Ширина",
            "lead_Линейная высота (см)",
            "lead_Вид оплаты",
            "lead_Служба доставки",
            "lead_Компания Отправитель",
            "lead_group_id",
            "lead_Масса (гр)",
            "lead_closed_dt",
            "contact_created_dt",
            "contact_updated_dt",
            "lead_utm_medium",
            "lead_Категория и варианты выбора",
            "received_dt",
            "lead_Модель телефона",
            "lead_Дата перехода Передан в доставку",
            "lead_created_dt",
            "handed_to_delivery_dt",
            "sale_date_dt",
            "lead_utm_campaign",
            "contact_Источник трафика",
            "lead_Дата перехода в Сборку",
            "lead_utm_term",
            "lead_utm_sky",
            "rejected_dt",
            "lead_group",
            "lead_Проблема",
            self.target_col,
        ]

    # =========================
    # public API
    # =========================

    def analyze(self, df: pd.DataFrame) -> AnalyzerResult:
        available = [c for c in self.required_columns if c in df.columns]
        missing = [c for c in self.required_columns if c not in df.columns]

        planned_new = [
            "width_cat",
            "lead_height_known",
            "lead_payment_type",
            "lead_delivery_type",
            "lead_group_quality",
            "lead_mass_known",
            "lead_mass_log",
            "contact_update_delay_log",
            "is_paid_traffic",
            "lead_category_freq",
            "is_feature_phone",
            "sale_to_delivery_log",
            "sale_to_delivery_missing",
            "lead_to_delivery_days_log",
            "lead_to_delivery_days_missing",
            "lead_utm_campaign_missing",
            "lead_utm_campaign_grouped",
            "traffic_source_missing",
            "traffic_source_grouped",
            "lead_to_assembly_days",
            "assembly_flag",
            "lead_to_assembly_long",
            "utm_term_missing",
            "utm_term_grouped",
            "utm_sky_missing",
            "utm_sky_autotarget",
            "utm_sky_brand",
            "utm_sky_varicose",
            "utm_sky_sleep",
            "lead_group_missing",
            "lead_group_grouped",
            "problem_missing",
            "problem_grouped",
        ]

        planned_drop = [
            "lead_Ширина",
            "lead_Линейная высота (см)",
            "lead_Вид оплаты",
            "returned_dt",
            "lead_Служба доставки",
            "lead_Компания Отправитель",
            "lead_group_id",
            "lead_Масса (гр)",
            "lead_closed_dt",
            "contact_created_dt",
            "contact_updated_dt",
            "lead_utm_medium",
            "lead_Категория и варианты выбора",
            "received_dt",
            "lead_Модель телефона",
            "lead_Дата перехода Передан в доставку",
            "lead_to_delivery_days",
            "sale_to_delivery_days",
            "handed_to_delivery_dt",
            "lead_utm_campaign",
            "contact_Источник трафика",
            "lead_Дата перехода в Сборку",
            "lead_utm_term",
            "lead_utm_sky",
            "rejected_dt",
            "lead_group",
            "lead_Проблема",
        ]

        stats = {
            "input_rows": int(len(df)),
            "input_columns": int(df.shape[1]),
            "required_columns_total": len(self.required_columns),
            "required_columns_available": len(available),
            "required_columns_missing": len(missing),
            "date_range_start": str(self.start.date()),
            "date_range_end": str(self.end.date()),
            "planned_new_columns_count": len(planned_new),
            "planned_drop_columns_count": len(planned_drop),
        }

        return AnalyzerResult(
            name=self.name,
            payload={
                "description": "Структурированный analyzer для feature engineering с контролем дат и отложенным удалением полей.",

                "columns": {
                    "required": self.required_columns.copy(),
                    "available": available,
                    "missing": missing,
                },

                "plan": {
                    "new_columns": planned_new,
                    "drop_columns": planned_drop,
                },

                "stats": stats,
            },
            meta={
                "rules": {
                    "delayed_drop": True,
                    "date_bounds_enabled": True,
                    "target_col": self.target_col,
                }
            }
        )

    def apply(
        self,
        df: pd.DataFrame,
        result: AnalyzerResult,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        work_df = df.copy()
        #self._validate_required_columns(work_df, result.required_columns)

        ctx = self._make_context(work_df)

        self._transform_width(work_df, ctx)
        self._transform_height(work_df, ctx)
        self._transform_payment_type(work_df, ctx)
        self._mark_drop(ctx, ["returned_dt"])
        self._transform_delivery_type(work_df, ctx)
        self._mark_drop(ctx, ["lead_Компания Отправитель"])
        self._transform_group_quality(work_df, ctx)
        self._transform_mass(work_df, ctx)
        self._mark_drop(ctx, ["lead_closed_dt"])
        self._transform_contact_update_delay(work_df, ctx)
        self._transform_paid_traffic(work_df, ctx)
        self._transform_category_freq(work_df, ctx)
        self._mark_drop(ctx, ["received_dt"])
        self._transform_phone_model(work_df, ctx)
        self._mark_drop(ctx, ["lead_Дата перехода Передан в доставку"])
        self._transform_delivery_delays(work_df, ctx)
        self._transform_utm_campaign(work_df, ctx)
        self._transform_traffic_source(work_df, ctx)
        self._transform_assembly_delay(work_df, ctx)
        self._transform_utm_term(work_df, ctx)
        self._transform_utm_sky(work_df, ctx)
        self._mark_drop(ctx, ["rejected_dt"])
        self._transform_lead_group(work_df, ctx)
        self._transform_problem_group(work_df, ctx)

        work_df = self._apply_deferred_drops(work_df, ctx)
        meta = self._finalize_meta(df, work_df, ctx)
        self._print_summary(meta)
        return work_df, meta

    # =========================
    # context / validation / stats
    # =========================

    def _make_context(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows_before": int(len(df)),
            "cols_before": int(df.shape[1]),
            "new_columns": [],
            "dropped_columns": [],
            "skipped_drops": [],
            "checks": [],
            "date_cleaning": {},
            "transforms": {},
        }

    def _validate_required_columns(self, df: pd.DataFrame, required_columns: list[str]) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise KeyError(
                "Отсутствуют обязательные поля для analyzer: " + ", ".join(missing)
            )

    def _assert_has_columns(self, df: pd.DataFrame, columns: list[str], step_name: str) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise KeyError(
                f"Шаг '{step_name}' не может быть выполнен. Нет полей: {', '.join(missing)}"
            )

    def _record_transform(self, ctx: dict[str, Any], name: str, payload: dict[str, Any]) -> None:
        ctx["transforms"][name] = payload

    def _record_new_columns(self, ctx: dict[str, Any], columns: list[str]) -> None:
        ctx["new_columns"].extend(columns)

    def _mark_drop(self, ctx: dict[str, Any], columns: list[str]) -> None:
        ctx["dropped_columns"].extend(columns)

    def _finalize_meta(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        ctx: dict[str, Any],
    ) -> dict[str, Any]:
        unique_new_columns = list(dict.fromkeys(ctx["new_columns"]))
        unique_dropped_columns = list(dict.fromkeys(ctx["dropped_columns"]))
        unique_skipped_drops = list(dict.fromkeys(ctx["skipped_drops"]))

        meta = {
            "analyzer": self.name,
            "rows_before": ctx["rows_before"],
            "rows_after": int(len(df_after)),
            "cols_before": ctx["cols_before"],
            "cols_after": int(df_after.shape[1]),
            "processed_existing_columns": int(df_before.shape[1]),
            "new_columns_count": len(unique_new_columns),
            "dropped_columns_count": len(unique_dropped_columns),
            "skipped_drop_columns_count": len(unique_skipped_drops),
            "new_columns": unique_new_columns,
            "dropped_columns": unique_dropped_columns,
            "skipped_drop_columns": unique_skipped_drops,
            "date_cleaning": ctx["date_cleaning"],
            "transforms": ctx["transforms"],
        }
        return meta

    def _print_summary(self, meta: dict[str, Any]) -> None:
        print("=" * 80)
        print(f"Analyzer: {meta['analyzer']}")
        print(f"Rows: {meta['rows_before']} -> {meta['rows_after']}")
        print(f"Cols: {meta['cols_before']} -> {meta['cols_after']}")
        print(f"Processed columns: {meta['processed_existing_columns']}")
        print(f"New columns: {meta['new_columns_count']}")
        print(f"Dropped columns: {meta['dropped_columns_count']}")
        print(f"Skipped drops: {meta['skipped_drop_columns_count']}")
        print("Date cleaning stats:")
        for col, info in meta["date_cleaning"].items():
            print(f"  - {col}: {info}")
        print("Transform stats:")
        for name, info in meta["transforms"].items():
            print(f"  - {name}: {info}")
        print("=" * 80)

    # =========================
    # common helpers
    # =========================

    def _safe_to_datetime(self, s: pd.Series) -> pd.Series:
        dt = pd.to_datetime(s, errors="coerce")
        if isinstance(dt.dtype, pd.DatetimeTZDtype):
            dt = dt.dt.tz_localize(None)
        return dt

    def _clip_datetime_range(
        self,
        df: pd.DataFrame,
        col: str,
        ctx: dict[str, Any],
    ) -> None:
        self._assert_has_columns(df, [col], f"clip_datetime_range::{col}")
        before_non_null = int(df[col].notna().sum())
        df[col] = self._safe_to_datetime(df[col])

        # Отключил пока
        #out_of_range_mask = df[col].notna() & ((df[col] < self.start) |
        # (df[col] > self.end))

        #out_of_range_count = int(out_of_range_mask.sum())
        #df.loc[out_of_range_mask, col] = pd.NaT

        out_of_range_count = 0
        after_non_null = int(df[col].notna().sum())
        ctx["date_cleaning"][col] = {
            "non_null_before": before_non_null,
            "out_of_range_set_to_nat": out_of_range_count,
            "non_null_after": after_non_null,
            "start": str(self.start.date()),
            "end": str(self.end.date()),
        }

    def _safe_log1p_nonnegative(self, s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        s = s.where(s >= 0)
        return np.log1p(s)

    def _top_k_grouped(
        self,
        s: pd.Series,
        top_k: int,
        other_label: str,
    ) -> pd.Series:
        top_values = s.value_counts(dropna=True).head(top_k).index
        return s.where(s.isin(top_values), other_label)

    def _apply_deferred_drops(self, df: pd.DataFrame, ctx: dict[str, Any]) -> pd.DataFrame:
        planned = list(dict.fromkeys(ctx["dropped_columns"]))
        existing = [col for col in planned if col in df.columns]
        skipped = [col for col in planned if col not in df.columns]
        ctx["skipped_drops"].extend(skipped)
        if existing:
            df = df.drop(columns=existing)
        return df

    # =========================
    # feature methods
    # =========================

    def _transform_width(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        self._assert_has_columns(df, ["lead_Ширина"], "width")

        def width_category(x: Any) -> str:
            if pd.isna(x):
                return "unknown"
            if x < 10 or x > 50:
                return "anomaly"
            if x < 25:
                return "small"
            if x <= 35:
                return "normal"
            return "large"

        df["width_cat"] = df["lead_Ширина"].apply(width_category)
        self._record_new_columns(ctx, ["width_cat"])
        self._mark_drop(ctx, ["lead_Ширина"])
        self._record_transform(
            ctx,
            "width",
            {"source": "lead_Ширина", "target": "width_cat", "nunique": int(df["width_cat"].nunique(dropna=False))},
        )

    def _transform_height(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        self._assert_has_columns(df, ["lead_Линейная высота (см)"], "height")
        df["lead_height_known"] = df["lead_Линейная высота (см)"].notna().astype(int)
        self._record_new_columns(ctx, ["lead_height_known"])
        self._mark_drop(ctx, ["lead_Линейная высота (см)"])
        self._record_transform(
            ctx,
            "height",
            {"known_share": round(float(df["lead_height_known"].mean()), 6)},
        )

    def _transform_payment_type(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        self._assert_has_columns(df, ["lead_Вид оплаты"], "payment_type")

        def payment_type(x: Any) -> str:
            if pd.isna(x):
                return "unknown"
            if x == "Наложенный платеж":
                return "cash_on_delivery"
            return "online"

        df["lead_payment_type"] = df["lead_Вид оплаты"].apply(payment_type)
        self._record_new_columns(ctx, ["lead_payment_type"])
        self._mark_drop(ctx, ["lead_Вид оплаты"])
        self._record_transform(
            ctx,
            "payment_type",
            {"value_counts": df["lead_payment_type"].value_counts(dropna=False).to_dict()},
        )

    def _transform_delivery_type(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        self._assert_has_columns(df, ["lead_Служба доставки"], "delivery_type")

        def delivery_type(x: Any) -> str:
            if pd.isna(x):
                return "unknown"
            if x in ["СДЭК до ПВЗ", "Самовывоз"]:
                return "pickup_point"
            if x in ["СДЭК до Двери", "Курьер ЕМС"]:
                return "door_delivery"
            if x == "Почта":
                return "post"
            return "unknown"

        df["lead_delivery_type"] = df["lead_Служба доставки"].apply(delivery_type)
        self._record_new_columns(ctx, ["lead_delivery_type"])
        self._mark_drop(ctx, ["lead_Служба доставки"])
        self._record_transform(
            ctx,
            "delivery_type",
            {"value_counts": df["lead_delivery_type"].value_counts(dropna=False).to_dict()},
        )

    def _transform_group_quality(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        self._assert_has_columns(df, ["lead_group_id", self.target_col], "group_quality")
        group_id = df["lead_group_id"].astype("object").fillna("unknown").astype("category")
        group_means = df.groupby(group_id, observed=False)[self.target_col].mean()
        df["lead_group_quality"] = group_id.map(group_means).astype(float)
        self._record_new_columns(ctx, ["lead_group_quality"])
        self._mark_drop(ctx, ["lead_group_id"])
        self._record_transform(
            ctx,
            "group_quality",
            {
                "groups": int(group_id.nunique(dropna=False)),
                "min_quality": float(df["lead_group_quality"].min()),
                "max_quality": float(df["lead_group_quality"].max()),
            },
        )

    def _transform_mass(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        self._assert_has_columns(df, ["lead_Масса (гр)"], "mass")
        df["lead_mass_known"] = df["lead_Масса (гр)"].notna().astype(int)
        df["lead_mass_log"] = self._safe_log1p_nonnegative(df["lead_Масса (гр)"]).fillna(0)
        self._record_new_columns(ctx, ["lead_mass_known", "lead_mass_log"])
        self._mark_drop(ctx, ["lead_Масса (гр)"])
        self._record_transform(
            ctx,
            "mass",
            {
                "known_share": round(float(df["lead_mass_known"].mean()), 6),
                "mass_log_mean": round(float(df["lead_mass_log"].mean()), 6),
            },
        )

    def _transform_contact_update_delay(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        required = ["contact_created_dt", "contact_updated_dt"]
        self._assert_has_columns(df, required, "contact_update_delay")

        for col in required:
            self._clip_datetime_range(df, col, ctx)

        delay_days = (df["contact_updated_dt"] - df["contact_created_dt"]).dt.total_seconds() / 86400
        delay_days = delay_days.where(delay_days >= 0)
        median_value = delay_days.median()
        if pd.isna(median_value):
            median_value = 0.0
        delay_days = delay_days.fillna(median_value)

        df["contact_update_delay_log"] = np.log1p(delay_days)
        self._record_new_columns(ctx, ["contact_update_delay_log"])
        self._mark_drop(ctx, ["contact_created_dt", "contact_updated_dt"])
        self._record_transform(
            ctx,
            "contact_update_delay",
            {
                "median_days_filled": round(float(median_value), 6),
                "delay_log_mean": round(float(df["contact_update_delay_log"].mean()), 6),
            },
        )

    def _transform_paid_traffic(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        self._assert_has_columns(df, ["lead_utm_medium"], "paid_traffic")
        pattern = r"cpc|cpm"
        df["is_paid_traffic"] = (
            df["lead_utm_medium"].astype(str).str.lower().str.contains(pattern, regex=True, na=False).astype(int)
        )
        self._record_new_columns(ctx, ["is_paid_traffic"])
        self._mark_drop(ctx, ["lead_utm_medium"])
        self._record_transform(
            ctx,
            "paid_traffic",
            {"share_paid": round(float(df["is_paid_traffic"].mean()), 6)},
        )

    def _transform_category_freq(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        col = "lead_Категория и варианты выбора"
        self._assert_has_columns(df, [col], "category_freq")
        cleaned = df[col].fillna("unknown").replace({"Нет категории": "unknown"})
        freq = cleaned.value_counts(normalize=True, dropna=False)
        df["lead_category_freq"] = cleaned.map(freq).astype(float)
        self._record_new_columns(ctx, ["lead_category_freq"])
        self._mark_drop(ctx, [col])
        self._record_transform(
            ctx,
            "category_freq",
            {"unique_categories": int(cleaned.nunique(dropna=False))},
        )

    def _transform_phone_model(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        col = "lead_Модель телефона"
        self._assert_has_columns(df, [col], "phone_model")
        cleaned = df[col].fillna("unknown")
        df["is_feature_phone"] = (cleaned == "Кнопочный").astype(int)
        self._record_new_columns(ctx, ["is_feature_phone"])
        self._mark_drop(ctx, [col])
        self._record_transform(
            ctx,
            "phone_model",
            {"share_feature_phone": round(float(df["is_feature_phone"].mean()), 6)},
        )

    def _transform_delivery_delays(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        required = ["lead_created_dt", "handed_to_delivery_dt", "sale_date_dt"]
        self._assert_has_columns(df, required, "delivery_delays")

        for col in required:
            self._clip_datetime_range(df, col, ctx)

        lead_to_delivery_days = (df["handed_to_delivery_dt"] - df["lead_created_dt"]).dt.total_seconds() / 86400
        sale_to_delivery_days = (df["handed_to_delivery_dt"] - df["sale_date_dt"]).dt.total_seconds() / 86400

        lead_to_delivery_days = lead_to_delivery_days.where(lead_to_delivery_days >= 0)
        sale_to_delivery_days = sale_to_delivery_days.where(sale_to_delivery_days >= 0)

        df["lead_to_delivery_days_log"] = np.log1p(lead_to_delivery_days)
        df["sale_to_delivery_log"] = np.log1p(sale_to_delivery_days)

        df["lead_to_delivery_days_missing"] = df["lead_to_delivery_days_log"].isna().astype(int)
        df["sale_to_delivery_missing"] = df["sale_to_delivery_log"].isna().astype(int)

        df["lead_to_delivery_days_log"] = df["lead_to_delivery_days_log"].fillna(-1)
        df["sale_to_delivery_log"] = df["sale_to_delivery_log"].fillna(-1)

        self._record_new_columns(
            ctx,
            [
                "lead_to_delivery_days_log",
                "lead_to_delivery_days_missing",
                "sale_to_delivery_log",
                "sale_to_delivery_missing",
            ],
        )
        self._mark_drop(ctx, ["handed_to_delivery_dt"])
        self._record_transform(
            ctx,
            "delivery_delays",
            {
                "lead_missing_share": round(float(df["lead_to_delivery_days_missing"].mean()), 6),
                "sale_missing_share": round(float(df["sale_to_delivery_missing"].mean()), 6),
            },
        )

    def _transform_utm_campaign(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        col = "lead_utm_campaign"
        self._assert_has_columns(df, [col], "utm_campaign")
        cleaned = df[col].replace(["{campaing_id}", "Неизвестно"], np.nan)
        df["lead_utm_campaign_missing"] = cleaned.isna().astype(int)
        df["lead_utm_campaign_grouped"] = self._top_k_grouped(cleaned, top_k=20, other_label="unknown")
        self._record_new_columns(ctx, ["lead_utm_campaign_missing", "lead_utm_campaign_grouped"])
        self._mark_drop(ctx, [col])
        self._record_transform(
            ctx,
            "utm_campaign",
            {
                "top_k": 20,
                "missing_share": round(float(df["lead_utm_campaign_missing"].mean()), 6),
                "grouped_unique": int(df["lead_utm_campaign_grouped"].nunique(dropna=False)),
            },
        )

    def _transform_traffic_source(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        col = "contact_Источник трафика"
        self._assert_has_columns(df, [col], "traffic_source")
        cleaned = df[col]
        df["traffic_source_missing"] = cleaned.isna().astype(int)
        df["traffic_source_grouped"] = self._top_k_grouped(cleaned, top_k=20, other_label="other")
        self._record_new_columns(ctx, ["traffic_source_missing", "traffic_source_grouped"])
        self._mark_drop(ctx, [col])
        self._record_transform(
            ctx,
            "traffic_source",
            {
                "top_k": 20,
                "missing_share": round(float(df["traffic_source_missing"].mean()), 6),
                "grouped_unique": int(df["traffic_source_grouped"].nunique(dropna=False)),
            },
        )

    def _transform_assembly_delay(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        required = ["lead_created_dt", "lead_Дата перехода в Сборку"]
        self._assert_has_columns(df, required, "assembly_delay")

        for col in required:
            self._clip_datetime_range(df, col, ctx)

        delay = (df["lead_Дата перехода в Сборку"] - df["lead_created_dt"]).dt.total_seconds() / 86400
        delay = delay + 1

        df["assembly_flag"] = df["lead_Дата перехода в Сборку"].notna().astype(int)
        df["lead_to_assembly_long"] = (delay > 30).astype(int)
        df["lead_to_assembly_days"] = delay.clip(upper=30).fillna(-1)

        self._record_new_columns(ctx, ["assembly_flag", "lead_to_assembly_long", "lead_to_assembly_days"])
        self._mark_drop(ctx, ["lead_Дата перехода в Сборку"])
        self._record_transform(
            ctx,
            "assembly_delay",
            {
                "assembly_share": round(float(df["assembly_flag"].mean()), 6),
                "long_share": round(float(df["lead_to_assembly_long"].mean()), 6),
            },
        )

    def _transform_utm_term(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        col = "lead_utm_term"
        self._assert_has_columns(df, [col], "utm_term")
        cleaned = df[col].replace(["Неизвестно"], np.nan)
        df["utm_term_missing"] = cleaned.isna().astype(int)
        df["utm_term_grouped"] = self._top_k_grouped(cleaned, top_k=40, other_label="other")
        self._record_new_columns(ctx, ["utm_term_missing", "utm_term_grouped"])
        self._mark_drop(ctx, [col])
        self._record_transform(
            ctx,
            "utm_term",
            {
                "top_k": 40,
                "missing_share": round(float(df["utm_term_missing"].mean()), 6),
                "grouped_unique": int(df["utm_term_grouped"].nunique(dropna=False)),
            },
        )

    def _transform_utm_sky(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        col = "lead_utm_sky"
        self._assert_has_columns(df, [col], "utm_sky")
        cleaned = df[col].replace(["{keyword}"], np.nan)
        as_str = cleaned.astype("string")

        df["utm_sky_missing"] = cleaned.isna().astype(int)
        df["utm_sky_autotarget"] = (cleaned == "---autotargeting").astype(int)
        df["utm_sky_brand"] = as_str.str.contains("artraid|артрейд", case=False, na=False, regex=True).astype(int)
        df["utm_sky_varicose"] = as_str.str.contains("варикоз|вены|флеболог", case=False, na=False, regex=True).astype(int)
        df["utm_sky_sleep"] = as_str.str.contains("сон|sleep", case=False, na=False, regex=True).astype(int)

        self._record_new_columns(
            ctx,
            [
                "utm_sky_missing",
                "utm_sky_autotarget",
                "utm_sky_brand",
                "utm_sky_varicose",
                "utm_sky_sleep",
            ],
        )
        self._mark_drop(ctx, [col])
        self._record_transform(
            ctx,
            "utm_sky",
            {
                "missing_share": round(float(df["utm_sky_missing"].mean()), 6),
                "autotarget_share": round(float(df["utm_sky_autotarget"].mean()), 6),
            },
        )

    def _transform_lead_group(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        col = "lead_group"
        self._assert_has_columns(df, [col], "lead_group")
        main_groups = ["yur", "but"]
        df["lead_group_missing"] = df[col].isna().astype(int)
        df["lead_group_grouped"] = df[col].where(df[col].isin(main_groups), "other")
        self._record_new_columns(ctx, ["lead_group_missing", "lead_group_grouped"])
        self._mark_drop(ctx, [col])
        self._record_transform(
            ctx,
            "lead_group",
            {
                "main_groups": main_groups,
                "missing_share": round(float(df["lead_group_missing"].mean()), 6),
            },
        )

    def _transform_problem_group(self, df: pd.DataFrame, ctx: dict[str, Any]) -> None:
        col = "lead_Проблема"
        self._assert_has_columns(df, [col], "problem_group")
        main_groups = [
            "Суставы и позвоночник",
            "Варикоз",
            "Сердечно-сосудистые заболевания",
            "Бессоница",
            "Головные боли",
            "Отеки",
            "Зрительная система",
            "Давление",
            "Инсульт",
            "Боли и тяжесть в ногах",
        ]
        df["problem_missing"] = df[col].isna().astype(int)
        df["problem_grouped"] = df[col].where(df[col].isin(main_groups), "other")
        self._record_new_columns(ctx, ["problem_missing", "problem_grouped"])
        self._mark_drop(ctx, [col])
        self._record_transform(
            ctx,
            "problem_group",
            {
                "main_groups_count": len(main_groups),
                "missing_share": round(float(df["problem_missing"].mean()), 6),
            },
        )


__all__ = [
    "AnalyzerResult",
    "BaseAnalyzer",
    "LeadFeatureEngineeringAnalyzer",
]
