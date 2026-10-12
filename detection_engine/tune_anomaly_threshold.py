
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed" / "binary_network_traffic.csv"
OUT = ROOT / "reports" / "anomaly_threshold_comparison.csv"

df = pd.read_csv(DATA, low_memory=False)
labels = df["Label"].astype(str).str.strip().str.upper()
valid = ~labels.isin(["", "NAN", "NONE", "NULL"])
df = df.loc[valid].reset_index(drop=True)
labels = labels.loc[valid].reset_index(drop=True)

y = (~labels.isin(["BENIGN", "BENIGNTRAFFIC", "NORMAL", "0"])).astype(int)
X = df.drop(columns=["Label", "binary_label"], errors="ignore")
X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Hold out some benign training records for threshold calibration.
benign = X_train.loc[y_train == 0]
fit_benign, calibration_benign = train_test_split(
    benign, test_size=0.15, random_state=123
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("isolation_forest", IsolationForest(
        n_estimators=250,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(fit_benign)

# Higher scores indicate more unusual traffic.
calibration_scores = -pipeline.decision_function(calibration_benign)
test_scores = -pipeline.decision_function(X_test)

rows = []
for percentile in [90, 95, 97.5, 99]:
    threshold = float(np.percentile(calibration_scores, percentile))
    pred = (test_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, pred, labels=[0, 1]
    ).ravel()

    rows.append({
        "calibration_percentile": percentile,
        "anomaly_threshold": threshold,
        "attack_precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "attack_recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "attack_f1": f1_score(y_test, pred, zero_division=0),
        "false_alarms": int(fp),
        "missed_attacks": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    })

results = pd.DataFrame(rows)
print("\n=== CALIBRATED ANOMALY THRESHOLDS ===")
print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

results.to_csv(OUT, index=False)
print("\nSaved:", OUT)
print("Original models and dashboard were not modified.")
