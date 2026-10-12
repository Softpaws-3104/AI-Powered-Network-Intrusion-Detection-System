#!/usr/bin/env python3
"""
Train and evaluate an Isolation Forest anomaly detector for
AI-Network-Traffic-Analyzer.

Run from the project root:
    python train_anomaly.py

The detector is trained only on BENIGN rows from the training split.
Attack labels are used only for evaluation, not for fitting the detector.
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "processed" / "binary_network_traffic.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "isolation_forest_v2.joblib"
REPORT_PATH = ROOT / "reports" / "isolation_forest_v2_test_results.csv"
METRICS_PATH = ROOT / "reports" / "isolation_forest_metrics.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CONTAMINATION = "auto"  # Threshold is learned by IsolationForest; tune after evaluation.

LABEL_CANDIDATES = (
    "Label", "label", "Class", "class", "Attack", "attack",
    "traffic_label", "Traffic_Label", "target", "Target"
)


def find_label_column(df: pd.DataFrame) -> str:
    # Prefer exact, common label names.
    for name in LABEL_CANDIDATES:
        if name in df.columns:
            return name

    # Fallback: case-insensitive matching.
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for name in ("label", "class", "attack", "traffic_label", "target"):
        if name in lookup:
            return lookup[name]

    raise ValueError(
        "Could not identify the target label column. "
        f"Available columns: {list(df.columns)}. "
        "Edit LABEL_CANDIDATES in train_anomaly.py to match your CSV."
    )


def normalize_labels(series: pd.Series) -> pd.Series:
    """Map benign labels to BENIGN and all other non-empty labels to ATTACK."""
    values = series.astype(str).str.strip().str.upper()
    benign = values.isin({"BENIGN", "BENIGNTRAFFIC", "NORMAL", "0"})
    # If the file uses numeric 0/1 labels, 0 is treated as BENIGN.
    result = pd.Series(np.where(benign, "BENIGN", "ATTACK"), index=series.index)
    missing = series.isna() | values.isin({"", "NAN", "NONE", "NULL"})
    result.loc[missing] = np.nan
    return result


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run this script from your project folder and verify "
            "processed/binary_network_traffic.csv exists."
        )

    print(f"[+] Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if df.empty:
        raise ValueError("The dataset is empty.")

    label_col = find_label_column(df)
    labels = normalize_labels(df[label_col])
    valid = labels.notna()
    df = df.loc[valid].copy()
    labels = labels.loc[valid]

    if labels.nunique() < 2:
        raise ValueError(
            f"Need both BENIGN and ATTACK labels for evaluation. "
            f"Values found in '{label_col}': {labels.value_counts().to_dict()}"
        )
     # Exclude both target-label columns from model inputs.
    label_columns = [
    col for col in ["Label", "binary_label"]
    if col in df.columns
    ]

    X = df.drop(columns=label_columns).copy()
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    print(f"[+] Excluded label columns: {label_columns}")
    print(f"[+] Selected feature count: {X.shape[1]}")


    # Drop entirely empty columns because median imputation cannot use them.
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        print(f"[!] Dropping all-empty numeric columns: {all_nan}")
        X = X.drop(columns=all_nan)

    # Drop constant columns; they provide no separation for this model.
    constant_cols = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
    if constant_cols:
        print(f"[!] Dropping constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)

    if X.shape[1] == 0:
        raise ValueError("No usable numeric feature columns were found.")

    # Split first, so the held-out test set is not used to fit the detector.
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    benign_train = X_train.loc[y_train == "BENIGN"]
    if len(benign_train) < 10:
        raise ValueError(
            f"Only {len(benign_train)} benign training rows were found; "
            "more benign samples are needed."
        )

    # Isolation Forest learns the normal/benign traffic profile only.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("isolation_forest", IsolationForest(
            n_estimators=250,
            max_samples="auto",
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print(f"[+] Total usable rows: {len(X):,}")
    print(f"[+] Numeric features: {X.shape[1]}")
    print(f"[+] Training rows: {len(X_train):,}; test rows: {len(X_test):,}")
    print(f"[+] Fitting only on benign training rows: {len(benign_train):,}")
    pipeline.fit(benign_train)

    # IsolationForest: predict 1 = inlier, -1 = anomaly.
    raw_prediction = pipeline.predict(X_test)
    y_pred = np.where(raw_prediction == -1, "ATTACK", "BENIGN")
    y_true = y_test.to_numpy()

    # Higher anomaly score means more unusual. Use the opposite of score_samples.
    anomaly_score = -pipeline.score_samples(X_test)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="ATTACK", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="ATTACK", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="ATTACK", zero_division=0)
    try:
        roc_auc = roc_auc_score((y_true == "ATTACK").astype(int), anomaly_score)
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=["BENIGN", "ATTACK"])
    report = classification_report(
        y_true, y_pred, labels=["BENIGN", "ATTACK"],
        target_names=["BENIGN", "ATTACK"], zero_division=0
    )

    print("\n========== ISOLATION FOREST RESULTS ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision (ATTACK): {precision:.4f}")
    print(f"Recall (ATTACK):    {recall:.4f}")
    print(f"F1 Score (ATTACK):  {f1:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted; BENIGN, ATTACK):")
    print(cm)
    print("\nClassification report:")
    print(report)

    # Save model metadata and the exact training feature order.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "label_column": label_col,
        "benign_label": "BENIGN",
        "attack_label": "ATTACK",
        "contamination": CONTAMINATION,
        "random_state": RANDOM_STATE,
        "description": "Isolation Forest fitted only on benign training traffic",
    }
    joblib.dump(model_bundle, MODEL_PATH)

    results = X_test.copy()
    results.insert(0, "true_label", y_test)
    results["predicted_label"] = y_pred
    results["anomaly_score"] = anomaly_score
    results["is_anomaly"] = (y_pred == "ATTACK")
    results.to_csv(REPORT_PATH, index=False)

    metrics_text = (
        "ISOLATION FOREST TEST RESULTS\n"
        f"Dataset: {DATA_PATH}\n"
        f"Label column: {label_col}\n"
        f"Rows evaluated: {len(X_test)}\n"
        f"Features: {X.shape[1]}\n"
        f"Benign training rows: {len(benign_train)}\n"
        f"Accuracy: {accuracy:.6f}\n"
        f"Precision_ATTACK: {precision:.6f}\n"
        f"Recall_ATTACK: {recall:.6f}\n"
        f"F1_ATTACK: {f1:.6f}\n"
        f"ROC_AUC: {roc_auc:.6f}\n\n"
        f"Confusion matrix (true rows / predicted columns; BENIGN, ATTACK):\n{cm}\n\n"
        f"{report}\n"
    )
    METRICS_PATH.write_text(metrics_text, encoding="utf-8")

    print(f"\n[+] Saved model: {MODEL_PATH}")
    print(f"[+] Saved test predictions: {REPORT_PATH}")
    print(f"[+] Saved metrics: {METRICS_PATH}")
    print("\nNext: review recall, false positives, and the confusion matrix before integrating this model into the dashboard.")


if __name__ == "__main__":
    main()
