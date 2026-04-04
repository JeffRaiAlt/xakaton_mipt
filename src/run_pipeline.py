from pathlib import Path
import pandas as pd

from feature_audit.feature_cleaning_pipeline_base import (
    FeatureCleaningPipeline)



DATA_PATH = Path("../data/raw/dataset_2025-03-01_2026-03-29_external.csv")
OUTPUT_DATA_PATH = Path("../data/raw/clean_dataset_step2.csv")
REPORT_PATH = Path("feature_cleaning_report.json")


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    # Очищаем и преобразовываем dataset не меняя число строк
    pipeline = FeatureCleaningPipeline()

    clean_df = pipeline.run(df, REPORT_PATH)
    print("\nFINAL SHAPE:", clean_df.shape)
    clean_df.to_csv(OUTPUT_DATA_PATH, index=False)

    print(f"Clean dataset saved to: {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    main()
