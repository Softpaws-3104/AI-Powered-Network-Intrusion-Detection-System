from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed" / "binary_network_traffic.csv"
MODEL_PATH = ROOT / "models" / "random_forest_binary.joblib"
REPORT = ROOT / "reports" / "rf_group_holdout_evaluation.txt"

df = pd.read_csv(DATA, low_memory=False)

if "Label" not in df.columns:
    raise ValueError("Dataset must contain a Label column.")

labels = df["Label"].astype(str).str.strip().str.upper()
benign_names = {"BENIGN", "BENIGNTRAFFIC", "NORMAL", "0", "0.0"}
valid = labels.notna() & ~labels.isin({"", "NAN", "NONE", "NULL"})
df = df.loc[valid].reset_index(drop=True)
labels = df["Label"].astype(str).str.strip().str.upper()
y = (~labels.isin(benign_names)).astype(int).to_numpy()

model = joblib.load(MODEL_PATH)
features = list(model.feature_names_in_)

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing model features: {missing}")

X = df[features].replace([np.inf, -np.inf], np.nan)

# Group identical feature vectors together. Hash collisions are checked
# by using the hash only as a grouping key, so review any unusual result.
groups = pd.util.hash_pandas_object(X, index=False).to_numpy()

# Count groups that have more than one original Label.
label_frame = pd.DataFrame({"group": groups, "label": labels.to_numpy()})
labels_per_group = label_frame.groupby("group")["label"].nunique()
conflicting_groups = int((labels_per_group > 1).sum())

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

train_groups = set(groups[train_idx])
test_groups = set(groups[test_idx])
overlap = len(train_groups.intersection(test_groups))

X_test = X.iloc[test_idx]
y_test = y[test_idx]

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Attack recall": recall_score(y_test, predictions, zero_division=0),
    "F1 score": f1_score(y_test, predictions, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, probabilities)
    if len(np.unique(y_test)) == 2 else float("nan"),
}

report_lines = [
    "RANDOM FOREST GROUP-AWARE HOLDOUT EVALUATION",
    f"Dataset rows: {len(df)}",
    f"Feature count: {len(features)}",
    f"Unique feature groups: {len(np.unique(groups))}",
    f"Training rows: {len(train_idx)}",
    f"Testing rows: {len(test_idx)}",
    f"Training feature groups: {len(train_groups)}",
    f"Testing feature groups: {len(test_groups)}",
    f"Train/test group overlap: {overlap}",
    f"Groups with conflicting original labels: {conflicting_groups}",
    "",
]

report_lines.extend(f"{name}: {value:.4f}" for name, value in metrics.items())
report_lines.extend([
    "",
    "Confusion matrix [[TN, FP], [FN, TP]]:",
    str([[int(tn), int(fp)], [int(fn), int(tp)]]),
    "",
    "Note: evaluates the saved Random Forest; no model retraining performed.",
])

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

print("\n".join(report_lines))
print(f"\nReport saved to: {REPORT}")
