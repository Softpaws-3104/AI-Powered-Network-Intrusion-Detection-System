
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed" / "binary_network_traffic.csv"
OUT = ROOT / "reports" / "fair_model_comparison.csv"
PRED_OUT = ROOT / "reports" / "fair_model_predictions.csv"

df = pd.read_csv(DATA, low_memory=False)

# Normalize one shared binary target.
labels = df["Label"].astype(str).str.strip().str.upper()
valid = ~labels.isin(["", "NAN", "NONE", "NULL"])
df = df.loc[valid].reset_index(drop=True)
labels = labels.loc[valid].reset_index(drop=True)

y = (~labels.isin(["BENIGN", "BENIGNTRAFFIC", "NORMAL", "0"])).astype(int)

# Exclude both label columns from all model inputs.
X = df.drop(columns=["Label", "binary_label"], errors="ignore")
X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

# Remove all-empty and constant columns.
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

# Create ONE shared stratified split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Features:", X.shape[1])
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Test labels:", y_test.value_counts().to_dict())

# Train a temporary Random Forest using the project's settings.
rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=2,
        random_state=42,
        class_weight="balanced",
    )),
])
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# Train a temporary Isolation Forest on benign training traffic only.
benign_train = X_train.loc[y_train == 0]
iforest = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", IsolationForest(
        n_estimators=250,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )),
])
iforest.fit(benign_train)

# Native Isolation Forest prediction: -1 = anomaly, 1 = normal.
if_pred = (iforest.predict(X_test) == -1).astype(int)

# A higher negated decision score means more anomalous.
anomaly_score = -iforest.decision_function(X_test)

# Evaluate both combined policies.
both_pred = ((rf_pred == 1) & (if_pred == 1)).astype(int)
either_pred = ((rf_pred == 1) | (if_pred == 1)).astype(int)

models = {
    "Random Forest": rf_pred,
    "Isolation Forest": if_pred,
    "Both models flag attack": both_pred,
    "Either model flags suspicious": either_pred,
}

rows = []
for name, pred in models.items():
    tn, fp, fn, tp = confusion_matrix(
        y_test, pred, labels=[0, 1]
    ).ravel()

    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "attack_precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "attack_recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "attack_f1": f1_score(y_test, pred, zero_division=0),
        "false_alarms": int(fp),
        "missed_attacks": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    })

metrics = pd.DataFrame(rows)
print("\n=== FAIR COMPARISON ===")
print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\nRandom Forest ROC-AUC:", round(
    roc_auc_score(y_test, rf_prob), 4
))

predictions = pd.DataFrame({
    "true_label": y_test.to_numpy(),
    "rf_prediction": rf_pred,
    "rf_attack_probability": rf_prob,
    "if_prediction": if_pred,
    "if_anomaly_score": anomaly_score,
    "both_attack": both_pred,
    "either_suspicious": either_pred,
})

OUT.parent.mkdir(parents=True, exist_ok=True)
metrics.to_csv(OUT, index=False)
predictions.to_csv(PRED_OUT, index=False)

print("\nSaved metrics:", OUT)
print("Saved predictions:", PRED_OUT)
print("Original model files and dashboard were not modified.")
