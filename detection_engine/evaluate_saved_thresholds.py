from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed" / "binary_network_traffic.csv"
MODEL = ROOT / "models" / "isolation_forest_v2.joblib"
OUT = ROOT / "reports" / "saved_anomaly_threshold_metrics.csv"

df = pd.read_csv(DATA, low_memory=False)

labels = df["Label"].astype(str).str.strip().str.upper()
valid = ~labels.isin(["", "NAN", "NONE", "NULL"])
df = df.loc[valid].reset_index(drop=True)
labels = labels.loc[valid].reset_index(drop=True)

y = (~labels.isin(
    ["BENIGN", "BENIGNTRAFFIC", "NORMAL", "0"]
)).astype(int)

bundle = joblib.load(MODEL)
pipeline = bundle["pipeline"]
features = bundle["feature_names"]

X = df[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

benign_train = X_train.loc[y_train == 0]
_, calibration_benign = train_test_split(
    benign_train, test_size=0.15, random_state=123
)

calibration_scores = -pipeline.decision_function(calibration_benign)
test_scores = -pipeline.decision_function(X_test)

rows = []

for percentile in [90, 95, 97.5, 99]:
    threshold = float(np.percentile(calibration_scores, percentile))
    predictions = (test_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    rows.append({
        "percentile": percentile,
        "threshold": threshold,
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "false_alarms": int(fp),
        "missed_attacks": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    })

results = pd.DataFrame(rows)
print(results.to_string(
    index=False, float_format=lambda value: f"{value:.4f}"
))

results.to_csv(OUT, index=False)
print(f"\nSaved evaluation: {OUT}")
print("No model or dashboard files modified.")
