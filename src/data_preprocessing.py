from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from feature_audit.selector.manual_feature_extraction import ManualFeatureExtractor
from feature_audit.selector.manual_feature_extraction import ManualFeatureExtractorConfig

INPUT_PATH = str(PROJECT_ROOT / "data" / "raw"
                  / "dataset_2025-03-01_2026-03-29_external.csv")

OUTPUT_PATH = str(PROJECT_ROOT / "data" / "cleaned_dataset" /"final_dataset.csv")

PVZ_LIST = str(PROJECT_ROOT / "data" / "raw"/ "PvzList_rus-2.xlsx")


final_features_list = [
    'lead_utm_group',
    'lead_utm_referrer',
    'lead_tags',
    'has_weight',
    'lead_utm_referrer_site',
    'lead_utm_id_1',
    'lead_utm_id_3',
    'lead_utm_device_type',
    'lead_creation_date_quarter',
    'lead_created_ts',
    'lead_length',
    'lead_payment_type',
    'lead_delivery_type',
    'lead_utm_campaign_missing',
    'lead_group_missing',
    'lead_height_bin',
    'delivery_cost_missing',
    'buyout_flag'
]

# 'sale_ts',
START_DATE = "2025-03-01"
END_DATE = "2026-03-29"
DATE_FILTER_COL = "lead_created_at"

def main():
    config = ManualFeatureExtractorConfig(
        data_path=INPUT_PATH,
        pvz_excel_path=PVZ_LIST
    )
    result_df = ManualFeatureExtractor(config=config).extract_features()
    df = result_df[final_features_list]
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Файл {OUTPUT_PATH} успешно создан.")

if __name__ == "__main__":
    main()


