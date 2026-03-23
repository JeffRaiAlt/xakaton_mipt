from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.assembler.assemble_dataset import assemble_dataset
from src.features.group_5.feature_processor import FeatureProcessor


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    processors = [
        FeatureProcessor(
            data_path=project_root / "data" / "raw" / "MIPT_hackathon_dataset.csv",
            feature_names_path=project_root / "data" / "feature_groups" / "features_group_5.txt",
            group_name="group_5",
            owner="member_5",
            specs_json_path=project_root / "src" / "features" / "group_5" / "example_feature_spec.json",
        ),
    ]

    final_dataset, feature_spec_df = assemble_dataset(
        processors=processors,
        output_dataset_path=project_root / "assembled_outputs" / "final_dataset.csv",
        output_feature_spec_path=project_root / "assembled_outputs" / "feature_spec.csv",
    )

    print(final_dataset.shape)
    print(feature_spec_df.head())


if __name__ == "__main__":
    main()
