
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detection_engine.unified_detector import UnifiedDetector

DATA_PATH = ROOT / "processed" / "binary_network_traffic.csv"
RESULT_PATH = ROOT / "reports" / "unified_detector_test_results.csv"

df = pd.read_csv(DATA_PATH)

# Use the same label convention as the existing training script.
if "Label" not in df.columns:
    raise ValueError("Expected Label column in the dataset.")

labels = df["Label"].astype(str).str.strip().str.upper()
y_true = (~labels.isin(["BENIGN", "NORMAL", "0", "0.0"])).astype(int)

# Reproduce the deterministic 20% test split used by train_anomaly.py.
from sklearn.model_selection import train_test_split

indices = df.index.to_numpy()
_, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y_true,
)

test_df = df.loc[test_indices].copy()
y_test = y_true.loc[test_indices].to_numpy()

detector = UnifiedDetector()
results = detector.predict(test_df)

# Evaluate only attacks requiring agreement from both models.
# Keep disagreements visible; do not silently count them as benign.
both_attack = results["unified_decision"].eq(
    "ATTACK_CONFIRMED_BY_BOTH"
)
any_suspicious = results["unified_decision"].ne(
    "BENIGN_CANDIDATE"
)

for name, prediction in [
    ("Random Forest", results["rf_prediction"].eq("ATTACK").astype(int)),
    ("Isolation Forest", results["is_anomaly"].astype(int)),
    ("Unified: both agree", both_attack.astype(int)),
    ("Unified: any suspicious", any_suspicious.astype(int)),
]:
    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy_score(y_test, prediction), 4))
    print("Attack precision:", round(
        precision_score(y_test, prediction, zero_division=0), 4
    ))
    print("Attack recall:", round(
        recall_score(y_test, prediction, zero_division=0), 4
    ))
    print("Attack F1:", round(
        f1_score(y_test, prediction, zero_division=0), 4
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, prediction, labels=[0, 1]))

try:
    auc = roc_auc_score(y_test, results["rf_attack_probability"])
    print("\nRandom Forest ROC-AUC:", round(auc, 4))
except ValueError as exc:
    print("ROC-AUC unavailable:", exc)

results.insert(0, "true_label", y_test)
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULT_PATH, index=False)

print("\nSaved results:", RESULT_PATH)
print("\nDecision distribution:")
print(results["unified_decision"].value_counts())
