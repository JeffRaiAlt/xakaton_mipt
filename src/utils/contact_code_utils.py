import pandas as pd
import re


def exctract_code_pvz(value):
    if pd.isna(value):
        return "unknown"
    match = re.fullmatch(r".*?([A-Z]+\d+).*", str(value))
    if match:
        return match.group(1)
    return "unknown"


def expand_cities_by_comma(pvz_data: pd.DataFrame) -> pd.DataFrame:
    expanded_rows = []
    for idx, row in pvz_data.iterrows():
        city = row["Город"]
        if "," in str(city):
            cities = [c.strip() for c in str(city).split(",")]
            for single_city in cities:
                new_row = row.copy()
                new_row["Город"] = single_city
                expanded_rows.append(new_row)
        else:
            expanded_rows.append(row)

    result_df = pd.DataFrame(expanded_rows)
    result_df = result_df.drop_duplicates()
    return result_df.reset_index(drop=True)


def get_region(code, pvz_dict):
    if pd.isna(code) or code == "unknown":
        return "unknown"
    code = str(code).strip()
    match = re.match(r"^([A-Z]+)", code)
    if match:
        code = match.group(1)
    else:
        return "unknown"
    region = pvz_dict.get(code, "unknown")
    if region == "unknown":
        return "unknown"
    return region
