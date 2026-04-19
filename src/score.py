import argparse
from pathlib import Path
import pandas as pd
import joblib
import feature_audit.selector.manual_feature_extraction as mfe
from utils.preprocess import evaluate

BASE_DIR = Path(__file__).resolve().parent.parent  # project/
MODEL_PATH = BASE_DIR / "model/model.pkl"

print("MODEL_PATH:", MODEL_PATH)

# =========================
# Args
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# =========================
# Load data
# =========================
config = mfe.ManualFeatureExtractorConfig(data_path=str(args.input))

# =========================
# Load models
# =========================
model = joblib.load(MODEL_PATH)

# =========================
# Features
# =========================
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
    'lead_length',
    'lead_payment_type',
    'lead_delivery_type',
    'lead_utm_campaign_missing',
    'lead_group_missing',
    'lead_height_bin',
    'delivery_cost_missing',
    'buyout_flag'
]

# здесь место трансформера
df = mfe.ManualFeatureExtractor(config=config).extract_features()

# =========================
# Predict
# =========================
TARGET = 'buyout_flag'

df = df[final_features_list].copy()
X = df.drop(columns=[TARGET])
y = df[TARGET]

probs_refuse = model.predict_proba(X)[:, 1]

print(evaluate(model, X, y))

# =========================
# Save output
# =========================
out = pd.DataFrame({
    "id": df.index,
    "score": probs_refuse
})


output_path = Path(args.output)

# создать директорию, если не существует
output_path.parent.mkdir(parents=True, exist_ok=True)

out.to_csv(output_path, index=False)