from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detection_engine.unified_detector import UnifiedDetector

DATA = ROOT / "processed" / "binary_network_traffic.csv"
OUTPUT = ROOT / "reports" / "unified_holdout_evaluation.csv"
METRICS_OUTPUT = ROOT / "reports" / "unified_holdout_metrics.csv"

df = pd.read_csv(DATA, low_memory=False)

if "Label" not in df.columns:
    raise ValueError("Dataset must contain a Label column.")

labels = df["Label"].astype(str).str.strip().str.upper()
valid = labels.isin(["BENIGN", "BENIGNTRAFFIC", "NORMAL", "0", "0.0"]) | (
    labels.notna() & ~labels.isin(["", "NAN", "NONE", "NULL"])
)

df = df.loc[valid].reset_index(drop=True)
labels = df["Label"].astype(str).str.strip().str.upper()
y = (~labels.isin(["BENIGN", "BENIGNTRAFFIC", "NORMAL", "0", "0.0"])).astype(int)

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=42,
    stratify=y,
)

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]
y_test = y.iloc[test_idx].to_numpy()

detector = UnifiedDetector(anomaly_percentile=95)

# Calibrate the score threshold from benign records in the training partition.
benign_reference = train_df.loc[
    y.iloc[train_idx].to_numpy() == 0,
    detector.features,
]
threshold = detector.calibrate(benign_reference)

results = detector.predict(test_df)
results.insert(0, "true_label", y_test)

rf_pred = results["rf_prediction"].eq("ATTACK").astype(int).to_numpy()
if_pred = results["is_anomaly"].astype(int).to_numpy()
both_pred = (
    results["unified_decision"] == "ATTACK_CONFIRMED_BY_BOTH"
).astype(int).to_numpy()
either_pred = (
    results["unified_decision"] != "BENIGN_CANDIDATE"
).astype(int).to_numpy()

policies = {
    "Random Forest": rf_pred,
    "Isolation Forest calibrated": if_pred,
    "Unified - both models agree": both_pred,
    "Unified - either model flags": either_pred,
}

metric_rows = []

for name, pred in policies.items():
    tn, fp, fn, tp = confusion_matrix(
        y_test, pred, labels=[0, 1]
    ).ravel()

    metric_rows.append({
        "policy": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, pred),
        "attack_precision": precision_score(y_test, pred, zero_division=0),
        "attack_recall": recall_score(y_test, pred, zero_division=0),
        "attack_f1": f1_score(y_test, pred, zero_division=0),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    })

metrics = pd.DataFrame(metric_rows)

try:
    rf_auc = roc_auc_score(y_test, results["rf_attack_probability"])
    print(f"Random Forest ROC-AUC: {rf_auc:.4f}")
except ValueError:
    print("Random Forest ROC-AUC unavailable.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)
metrics.to_csv(METRICS_OUTPUT, index=False)

print(f"\nHeld-out records: {len(test_df):,}")
print(f"Calibrated anomaly threshold: {threshold:.6f}")
print("\n=== Evaluation metrics ===")
print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nDecision distribution:")
print(results["unified_decision"].value_counts().to_string())
print(f"\nPredictions saved to: {OUTPUT}")
print(f"Metrics saved to: {METRICS_OUTPUT}")
