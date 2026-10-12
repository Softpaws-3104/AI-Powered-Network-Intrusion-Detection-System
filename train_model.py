import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path.home() / "AI-Network-Traffic-Analyzer"

DATA_FILE = BASE_DIR / "processed" / "binary_network_traffic.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "random_forest_binary.joblib"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")

# --------------------------------------------------
# FEATURES / TARGET
# --------------------------------------------------

# Remove columns that should NOT be model features
X = df.drop(columns=["Label", "binary_label"])

y = df["binary_label"]

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]:,}")

# --------------------------------------------------
# CLEAN NUMERIC FEATURES
# --------------------------------------------------

import numpy as np

print("\nChecking numeric features...")

# Convert infinity values to NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Count remaining missing values
missing_before = X.isna().sum().sum()

print(f"Missing/infinite values found: {missing_before}")

# Replace missing values with column medians
if missing_before > 0:
    X = X.fillna(X.median(numeric_only=True))

# Final safety check
remaining_bad = (
    ~np.isfinite(X.to_numpy())
).sum()

if remaining_bad > 0:
    raise ValueError(
        f"Dataset still contains {remaining_bad} "
        "non-finite values."
    )

print("Numeric feature check: PASSED")

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split:")
print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    n_jobs=2,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Training complete.")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "ATTACK"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(importance.head(15).to_string(index=False))

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(model, MODEL_FILE)

print("\nModel saved:")
print(MODEL_FILE)

# Save feature importance
importance.to_csv(
    MODEL_DIR / "feature_importance.csv",
    index=False
)

print("Feature importance saved.")
