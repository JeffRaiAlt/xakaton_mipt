from pathlib import Path
import pandas as pd
import json
from dataclasses import fields
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.feature_spec import FeatureSpec


def convert_feature_spec_csv_to_json(csv_path, json_path):
    df = pd.read_csv(csv_path)

    # dataclass
    required = [f.name for f in fields(FeatureSpec)]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    features = []

    for _, row in df.iterrows():
        features.append({
            field: row[field]
            for field in required
        })

    result = {"features": features}

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved:", json_path)

def create_feature_spec_template(X_block: pd.DataFrame) -> pd.DataFrame:
    """
    Создает шаблон feature_spec.csv:
    - name заполняется из X_block
    - остальные поля пустые (под ручное заполнение)
    """

    # все поля из dataclass
    columns = [f.name for f in fields(FeatureSpec)]

    # создаем DataFrame
    feature_spec = pd.DataFrame(columns=columns)

    # заполняем name
    feature_spec["name"] = X_block.columns

    # остальные поля оставляем пустыми
    for col in columns:
        if col != "name":
            feature_spec[col] = ""

    return feature_spec