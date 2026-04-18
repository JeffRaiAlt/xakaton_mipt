import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import openpyxl

import numpy as np
import pandas as pd
from utils import process_lead_utm


@dataclass
class ManualFeatureExtractorConfig:
    data_path: str
    target: str = "buyout_flag"
    pvz_excel_path: Optional[str | Path] = None
    start_date: str | None = None
    end_date: str | None = None
    date_filter_col: str | None = None


class ManualFeatureExtractor:

    def __init__(self, config: ManualFeatureExtractorConfig) -> None:
        self.config = config

    def extract_features(self) -> pd.DataFrame:
        df = self._load_data()
        return self._transform(df)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = self.transform_height_features(df)
        df = self.transform_payment_type(df)
        df = self.transform_delivery_type(df)
        df = self.transform_lead_group_grouped(df)
        df = self.transform_delivery_cost_features(df)

        df = self.transform_lead_utm_group(df)
        df = self.transform_lead_utm_referrer_site(df)
        df = self.transform_lead_tags(df)

        df = self.transform_lead_creation_date_features(df)

        df = self.transform_timedelta_and_created_features(df)
        df = self.transform_length_feature(df)
        #df = self.transform_mass(df)
        df = self.transform_utm_campaign(df)
        df = self.transform_wight(df)
        #df = self.transform_cluster(df)
        df = self.transform_utm_content_chain(df)

        # В конце!!!
        df = self.transform_timedelta(df)
        return df

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path, low_memory=False)

        # таргет
        df = df[df[self.config.target].notna()].copy()
        df[self.config.target] = df[self.config.target].astype(int)

        # даты
        if self.config.date_filter_col:
            df = self._prepare_datetime_column(df, self.config.date_filter_col)
            df = self._apply_date_filter(df)
            self._debug_datetime(df, self.config.date_filter_col)
        return df

    def _debug_datetime(self, df: pd.DataFrame, col: str):
        print("dtype:", df[col].dtype)
        print("min:", df[col].min())
        print("max:", df[col].max())
        print("n_na:", df[col].isna().sum())

    def _prepare_datetime_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        self._column_exists(df, col)

        # корректный парс unix timestamp
        df[col] = pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")

        # приводим к naive UTC (убираем tz, но сохраняем момент времени)
        df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)

        return df

    def _apply_date_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.config.date_filter_col

        if col not in df.columns:
            return df

        if self.config.start_date is not None:
            df = df[df[col] >= pd.Timestamp(self.config.start_date)]

        if self.config.end_date is not None:
            df = df[df[col] <= pd.Timestamp(self.config.end_date)]

        return df

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _column_exists(df: pd.DataFrame, col: str) -> bool:
        return col in df.columns

    @staticmethod
    def _extract_pvz_code(value: object) -> object:
        if pd.isna(value):
            return value
        match = re.fullmatch(r".*?([A-Z]+\d+).*", str(value))
        if match:
            return match.group(1)
        return value

    @staticmethod
    def _replace_non_matching_pvz(series: pd.Series) -> pd.Series:
        pattern = re.compile(r"^([A-Z]+)\d+$")
        result = []
        for value in series:
            if pd.isna(value):
                result.append("unknown")
            else:
                match = pattern.match(str(value).strip())
                if match:
                    result.append(match.group(1))
                else:
                    result.append("unknown")
        return pd.Series(result, index=series.index)

    def _load_pvz_dict(self) -> dict[str, str]:
        if self.config.pvz_excel_path is None:
            raise ValueError(
                "Для contact_region_pvz нужен pvz_excel_path с Excel-справочником ПВЗ."
            )

        pvz_data = pd.read_excel(
            self.config.pvz_excel_path,
            sheet_name="Россия",
            usecols=[0, 1, 2],
            engine="openpyxl",
        )

        expanded_rows = []
        for _, row in pvz_data.iterrows():
            city = row["Город"]
            if "," in str(city):
                cities = [c.strip() for c in str(city).split(",")]
                for single_city in cities:
                    new_row = row.copy()
                    new_row["Город"] = single_city
                    expanded_rows.append(new_row)
            else:
                expanded_rows.append(row)

        pvz_data = pd.DataFrame(expanded_rows).drop_duplicates().reset_index(drop=True)

        pvz_dict: dict[str, str] = {}
        for pvz_code, region in zip(pvz_data.iloc[:, 2], pvz_data.iloc[:, 0]):
            match = re.match(r"^([A-Z]+)", str(pvz_code))
            if match:
                pvz_dict[match.group(1)] = region
        return pvz_dict

    @staticmethod
    def _safe_to_datetime(series: pd.Series, unit: Optional[str] = None) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", unit=unit)

    @staticmethod
    def _split_utm_content(series: pd.Series) -> pd.DataFrame:
        def parse_utm(value: object) -> list[str]:
            value = str(value)
            parts = value.split("_")
            if len(parts) == 10:
                return parts
            return ["unknown"] * 10

        split_data = series.apply(parse_utm).apply(pd.Series)
        split_data.columns = [f"utm_{i + 1}" for i in range(10)]
        return split_data

    @staticmethod
    def _get_string_values(df: pd.DataFrame, column: str) -> list:
        unique_values = df[column].dropna().unique()
        numeric_mask = pd.to_numeric(pd.Series(unique_values), errors="coerce").notna()
        return list(pd.Series(unique_values)[~numeric_mask])

    # ------------------------------------------------------------------
    # feature-specific transforms
    # ------------------------------------------------------------------

    def transform_utm_campaign(self, df: pd.DataFrame) -> pd.DataFrame:
        df["lead_utm_campaign"] = (df["lead_utm_campaign"].replace
                                   (["{campaing_id}", "Неизвестно"],
                                    np.nan))
        # Признак сильно разреженный, отмечаем где не заполнен
        df["lead_utm_campaign_missing"] = df[
        "lead_utm_campaign"].isna().astype(int)
        return df

    def transform_height_features(self, df: pd.DataFrame) -> pd.DataFrame:
        col = "lead_Высота"
        if not self._column_exists(df, col):
            return df

        df["lead_height_known"] = df[col].notna().astype(int)

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 2 * iqr
        lead_height_clean = df[col].clip(upper=upper)

        def height_bin(x: object) -> str:
            if pd.isna(x):
                return "unknown"
            if x <= 4:
                return "small"
            if x <= 12:
                return "medium"
            return "large"

        df["lead_height_bin"] = lead_height_clean.apply(height_bin)
        df["lead_Высота"] = df["lead_Высота"].fillna(-1)
        return df

    def transform_wight(self, df: pd.DataFrame) -> pd.DataFrame:
        df["has_weight"] = np.where(df["lead_Вес (грамм)*"].isna(), 0, 1)
        return df

    def transform_payment_type(self, df: pd.DataFrame) -> pd.DataFrame:
        source_col = "lead_Вид оплаты"
        if not self._column_exists(df, source_col):
            return df

        def payment_type(x: object) -> str:
            if pd.isna(x):
                return "unknown"
            if x == "Наложенный платеж":
                return "cash_on_delivery"
            return "online"

        df["lead_payment_type"] = df[source_col].apply(payment_type)
        return df

    def transform_delivery_type(self, df: pd.DataFrame) -> pd.DataFrame:
        source_col = "lead_Служба доставки"
        if not self._column_exists(df, source_col):
            return df

        def delivery_type(x: object) -> str:
            if pd.isna(x):
                return "unknown"
            if x == "СДЭК до ПВЗ":
                return "pickup_point"
            if x in ["СДЭК до Двери", "Курьер ЕМС"]:
                return "door_delivery"
            if x == "Почта":
                return "post"
            if x == "Самовывоз":
                return "pickup_point"
            return "other"

        df["lead_delivery_type"] = df[source_col].apply(delivery_type)
        return df

    def transform_lead_group_grouped(self, df: pd.DataFrame) -> pd.DataFrame:
        col = "lead_group"
        if not self._column_exists(df, col):
            return df

        # флаг пропуска
        df["lead_group_missing"] = df["lead_group"].isna().astype(int)

        main_groups = ["yur", "but"]
        df["lead_group_grouped"] = df[col].where(df[col].isin(main_groups), "other")
        return df

    def transform_delivery_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        col = "lead_Стоимость доставки"
        if self._column_exists(df, col):
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"-": None, "": None})
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df["delivery_cost_missing"] = df[col].isna().astype(int)

            df["lead_shipping_cost"] = df[col].fillna(-1)

        elif self._column_exists(df, "lead_shipping_cost"):
            df["delivery_cost_missing"] = df["lead_shipping_cost"].isna().astype(int)
            df["lead_shipping_cost"] = df["lead_shipping_cost"].fillna(-1)

        df["lead_Стоимость доставки"] = df["lead_Стоимость доставки"].fillna(-1)

        return df

    def transform_lead_utm_group(self, df: pd.DataFrame) -> pd.DataFrame:
        col = "lead_utm_group"
        if not self._column_exists(df, col):
            return df

        def clean_utm_group(value: object) -> str:
            if not isinstance(value, str) or not value:
                return "unknown"
            match = re.search(r"[^A-Za-zА-Яа-яЁё]", value)
            if match:
                cleaned = value[:match.start()]
                return cleaned if cleaned else "unknown"
            return value

        df[col] = df[col].apply(clean_utm_group)
        top_5 = (
            df[df[col] != "unknown"][col]
            .value_counts()
            .head(5)
            .index
            .tolist()
        )
        mask = ~df[col].isin(top_5) & (df[col] != "unknown")
        df.loc[mask, col] = "rare_utm_group"
        df[col] = df[col].fillna("unknown")
        return df





    def transform_lead_utm_referrer_site(self, df: pd.DataFrame) -> pd.DataFrame:
        col = "lead_utm_referrer"
        if not self._column_exists(df, col):
            return df

        def extract_category(value):
            categories = ["varikoz", "otek", "sustav", "sleep", "davlenie"]
            # Обработка пропусков и нестроковых значений
            if not isinstance(value, str) or not value:
                return "unknown"

            # Приводим к нижнему регистру для корректного поиска
            value_lower = value.lower()

            # Ищем первую подходящую категорию в списке
            for category in categories:
                if category in value_lower:
                    return category

            # Если ни одна категория не найдена — возвращаем 'other'
            return "other"

        def get_site_category(url: object) -> str:
            if pd.isna(url) or not url:
                return "unknown"
            url_lower = str(url).lower()
            if "artraid" in url_lower:
                return "artraid"
            if "npotpz" in url_lower:
                return "npotpz"
            return "other"

        df["lead_utm_referrer_site"] = df[col].apply(get_site_category)

        df["lead_utm_referrer"] = df[
            "lead_utm_referrer"].apply(
            lambda x: x.rsplit("/", 1)[1] if isinstance(x,
                                                        str) and x else x
        )

        df["lead_utm_referrer"] = df[
            "lead_utm_referrer"].apply(extract_category)

        return df

    def transform_lead_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        col = "lead_tags"
        if not self._column_exists(df, col):
            return df

        categories = ["tilda", "npotpz", "Callibri", "ВХОДЯЩИЙ", "artraid"]

        def categorize_tags(value: object) -> str:
            if not isinstance(value, str) or not value:
                return "unknown"
            value_lower = value.lower()
            for category in categories:
                if category.lower() in value_lower:
                    return category
            return "unknown"

        df[col] = df[col].apply(categorize_tags)
        return df


    def transform_utm_content_chain(self,
                                    df: pd.DataFrame) -> pd.DataFrame:

        return process_lead_utm.add_lead_utm_device_type(df)

    def transform_utm_content_chain2(self,
                                    df: pd.DataFrame) -> pd.DataFrame:
        source_col = "lead_utm_content"
        utm_parsed = self._split_utm_content(df[source_col])
        df = pd.concat([df, utm_parsed], axis=1)

        res = self.get_string_values(df, "utm_1")

        for col in ["utm_1", "utm_2", "utm_3", "utm_8", "utm_9"]:
            df[col] = df[col].astype("string")
            df.loc[df[col].isin(res), col] = "-1"

        for col in ["utm_1", "utm_2", "utm_3", "utm_8", "utm_9"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
                -1).astype("int")

        df = df.rename(columns={
            "utm_1": "lead_utm_id_1",
            "utm_3": "lead_utm_id_3",

        })

        return df


    def transform_lead_creation_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        source_candidates = ["lead_Дата создания сделки", "lead_created_at"]
        source_col = next((c for c in source_candidates if c in df.columns), None)
        if source_col is None:
            return df

        if source_col == "lead_created_at":
            dt = self._safe_to_datetime(df[source_col], unit="s")
        else:
            dt = self._safe_to_datetime(df[source_col], unit="s")

        df["lead_creation_date_month"] = dt.dt.month.astype(
            "Int64").fillna(-1)
        df["lead_creation_date_quarter"] = dt.dt.quarter.astype(
            "Int64").fillna(-1)

        day = dt.dt.day
        df["lead_creation_date_sin"] = np.sin(2 * np.pi * day / 30)
        df["lead_creation_date_sin"] = df["lead_creation_date_sin"].fillna(-1)
        return df

    def transform_timedelta(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sale_ts"] = self._safe_to_datetime(
            df["sale_ts"], unit="s")
        return df

    def transform_timedelta_and_created_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not {"sale_ts", "lead_created_at"}.issubset(df.columns):
            return df

        # В ноутбуке дельта считалась как разница исходных секундовых ts
        sale_ts_num = pd.to_numeric(df["sale_ts"], errors="coerce")
        lead_created_num = pd.to_numeric(df["lead_created_at"], errors="coerce")
        df["timedelta_between_sale_and_creation"] = sale_ts_num - lead_created_num

        df["lead_created_ts"] = self._safe_to_datetime(df["lead_created_at"], unit="s")
        df["lead_created_dayofweek"] = df[
            "lead_created_ts"].dt.dayofweek.astype("int8").fillna(-1)
        return df

    def transform_length_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        source_col = "lead_Длина"
        if not self._column_exists(df, source_col):
            return df

        df["lead_length"] = df[source_col].fillna(-1)
        return df


    def transform_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        df["lead_tags_str"] = df["lead_tags"].apply(
            lambda x: "__".join(sorted(x)) if isinstance(x, list)
            else ("missing" if pd.isna(x) else str(x))
        )

        df["utm_combo"] = (
                df["lead_tags_str"] + "__" +
                df["lead_utm_referrer_site"].astype(str) + "__" +
                df["lead_utm_device_type"].astype(str)
        )


        df["bad_segment"] = (df["width_cat"] == "unknown").astype(int)


        """is_bad_segment = (
                (lead_tags == "tilda") &
                (referrer == "artraid") &
                (device == "unknown")
        )"""

        df = df.drop(columns=["lead_tags_str"])
        return df
