from pathlib import Path
import sys
import pandas as pd
import joblib

# =========================
# PATH SETUP
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# =========================
# IMPORTS
# =========================
from src.feature_audit.selector.manual_feature_extraction import (
    ManualFeatureExtractor,
    ManualFeatureExtractorConfig
)

from src.utils.preprocess import evaluate

# =========================
# PATHS
# =========================
INPUT_PATH = PROJECT_ROOT / "data/raw/dataset_2025-03-01_2026-03-29_external.csv"
PVZ_LIST = PROJECT_ROOT / "data/raw/PvzList_rus-2.xlsx"
MODEL_PATH = PROJECT_ROOT / "model/model.pkl"

# =========================
# FEATURES (must match training!)
# =========================
TARGET = "buyout_flag"

final_features_list = [
    'lead_responsible_user_id',
    'lead_utm_group',
    'lead_tags',
    'contact_region_pvz',
    'lead_utm_referrer_site',
    'lead_utm_id_1',
    'lead_utm_id_2',
    'lead_utm_id_3',
    'lead_utm_device_type',
    'lead_utm_reatrgeting_id',
    'lead_manager_category',
    'lead_creation_date_month',
    'lead_creation_date_quarter',
    'lead_creation_date_sin',
    'sale_date_quarter',
    'sale_date_dayofweek',
    'lead_total_cost_from_composition',
    'lead_has_brace',
    'lead_discount_category',
    'sale_hour',
    'sale_month',
    'lead_source',
    'timedelta_between_sale_and_creation',
    'lead_created_ts',
    'lead_created_dayofweek',
    'lead_shipping_cost',
    'lead_length',
    'lead_price',
    'lead_group_id',
    'width_cat',
    'width_is_missing',
    'lead_payment_type',
    'lead_delivery_type',
    'contact_to_lead_hours',
    'contact_hour',
    'contact_month',
    'utm_sky_autotarget',
    'utm_sky_brand',
    'lead_group_grouped',
    'lead_height_known',
    'lead_height_bin',
    'delivery_cost_missing',
    'lead_Стоимость доставки',
    'lead_Масса (гр)',
    'lead_Высота',
    'sale_ts',
    'buyout_flag'
]

# =========================
# MAIN PIPELINE
# =========================
def main():
    print("🔄 Running feature extraction...")

    config = ManualFeatureExtractorConfig(
        data_path=str(INPUT_PATH),
        pvz_excel_path=str(PVZ_LIST)
    )

    df = ManualFeatureExtractor(config=config).extract_features()

    print("✅ Feature extraction done")

    # =========================
    # Prepare data
    # =========================
    df = df[final_features_list].copy()
    df.to_csv(str(PROJECT_ROOT / "bench" / "incorrect_dataset.csv"))

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"Data shape: {X.shape}")

    # =========================
    # Load model
    # =========================
    model = joblib.load(MODEL_PATH)

    print("✅ Model loaded")

    # =========================
    # Evaluate
    # =========================
    metrics = evaluate(model, X, y)

    print("\n📊 METRICS:")
    print(metrics)


if __name__ == "__main__":
    main()