import numpy as np
import pandas as pd


def split_utm_content(series):
    def parse_utm(value):
        value = str(value)
        parts = value.split("_")
        if len(parts) == 10:
            return parts
        else:
            return ["unknown"] * 10

    split_data = series.apply(parse_utm).apply(pd.Series)
    split_data.columns = [f"utm_{i + 1}" for i in range(10)]
    return split_data

def add_lead_utm_device_type(df: pd.DataFrame) -> pd.DataFrame:
    utm_parsed = split_utm_content(df["lead_utm_content"])

    df = df.copy()
    df["lead_utm_device_type"] = utm_parsed["utm_5"]
    df["lead_utm_id_1"] = utm_parsed["utm_1"]
    df["lead_utm_id_3"] = utm_parsed["utm_3"]

    df["lead_utm_device_type"] = df["lead_utm_device_type"].fillna("unknown")
    df["lead_utm_id_1"] = (
        pd.to_numeric(df["lead_utm_id_1"], errors="coerce")
        .fillna(-1)
        .astype("int")
    )
    df["lead_utm_id_3"] = np.where(
        utm_parsed["utm_3"].isin(["unknown", "undefined", ""]), -1,
        utm_parsed["utm_3"]
    ).astype("int")

    df.loc[
        df["lead_utm_device_type"].isin(["undefined", ""]),
        "lead_utm_device_type",
    ] = "unknown"

    return df