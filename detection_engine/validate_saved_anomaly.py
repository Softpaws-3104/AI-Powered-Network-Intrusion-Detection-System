from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed" / "binary_network_traffic.csv"
MODEL = ROOT / "models" / "isolation_forest_v2.joblib"

df = pd.read_csv(DATA, low_memory=False)

labels = df["Label"].astype(str).str.strip().str.upper()
y = (~labels.isin(["BENIGN", "BENIGNTRAFFIC", "NORMAL", "0"])).astype(int)

bundle = joblib.load(MODEL)
pipeline = bundle["pipeline"]
features = bundle["feature_names"]

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Missing model features: {missing}")

X = df[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Recreate the shared held-out split.
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Use training data only to calibrate the threshold.
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

benign_train = X_train.loc[y_train == 0]
_, calibration_benign = train_test_split(
    benign_train, test_size=0.15, random_state=123
)

calibration_scores = -pipeline.decision_function(calibration_benign)
test_scores = -pipeline.decision_function(X_test)

print("Saved model:", MODEL)
print("Feature count:", len(features))
print("Test rows:", len(X_test))
print("Calibration rows:", len(calibration_benign))
print("Calibration score range:",
      float(calibration_scores.min()),
      "to", float(calibration_scores.max()))
print("Test score range:",
      float(test_scores.min()),
      "to", float(test_scores.max()))

print("\nThresholds from the SAVED model:")
for percentile in [90, 95, 97.5, 99]:
    threshold = float(np.percentile(calibration_scores, percentile))
    flagged = int((test_scores > threshold).sum())
    print(
        f"{percentile:>5}% | threshold={threshold:.6f} "
        f"| flagged={flagged}"
    )

print("\nValidation complete. No model or dashboard files modified.")

