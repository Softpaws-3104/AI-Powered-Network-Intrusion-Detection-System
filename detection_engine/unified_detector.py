
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

RF_MODEL_PATH = ROOT / "models" / "random_forest_binary.joblib"
IF_MODEL_PATH = ROOT / "models" / "isolation_forest_v2.joblib"


class UnifiedDetector:
    """Combine supervised classification with anomaly detection."""

    def __init__(self, anomaly_percentile=95):
        self.rf_model = joblib.load(RF_MODEL_PATH)
        self.if_bundle = joblib.load(IF_MODEL_PATH)
        self.if_pipeline = self.if_bundle["pipeline"]

        self.features = list(self.if_bundle["feature_names"])
        self.anomaly_percentile = float(anomaly_percentile)

        rf_features = list(self.rf_model.feature_names_in_)
        if rf_features != self.features:
            raise ValueError(
                "Random Forest and Isolation Forest feature "
                "names or order do not match."
            )

        if not 0 < self.anomaly_percentile < 100:
            raise ValueError("Anomaly percentile must be between 0 and 100.")

        self.anomaly_threshold = None

    def calibrate(self, benign_reference):
        """
        Set an anomaly-score threshold using a separate, benign
        reference dataset. Higher anomaly scores mean more unusual.
        """
        missing = set(self.features) - set(benign_reference.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        X = benign_reference[self.features].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        decision_scores = self.if_pipeline.decision_function(X)
        anomaly_scores = -np.asarray(decision_scores)

        self.anomaly_threshold = float(
            np.percentile(anomaly_scores, self.anomaly_percentile)
        )
        return self.anomaly_threshold

    def predict(self, traffic):
        """Return per-record predictions and model scores."""
        missing = set(self.features) - set(traffic.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        X = traffic[self.features].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        probabilities = self.rf_model.predict_proba(X)
        classes = list(self.rf_model.classes_)
        attack_index = classes.index(1)
        attack_probability = probabilities[:, attack_index]

        # Isolation Forest decision_function: lower = more anomalous.
        # Negating it makes higher anomaly scores more unusual.
        anomaly_scores = -np.asarray(
            self.if_pipeline.decision_function(X)
        )

        # Use the model's native prediction until calibrated.
        if self.anomaly_threshold is None:
            anomaly_prediction = (
                self.if_pipeline.predict(X) == -1
            )
        else:
            anomaly_prediction = (
                anomaly_scores > self.anomaly_threshold
            )

        rf_attack = attack_probability >= 0.5

        # Preserve disagreements for review rather than hiding them.
        decision = np.select(
            [
                rf_attack & anomaly_prediction,
                rf_attack & ~anomaly_prediction,
                ~rf_attack & anomaly_prediction,
            ],
            [
                "ATTACK_CONFIRMED_BY_BOTH",
                "ATTACK_RF_ONLY",
                "SUSPICIOUS_ANOMALY_ONLY",
            ],
            default="BENIGN_CANDIDATE",
        )

        return pd.DataFrame({
            "rf_attack_probability": attack_probability,
            "rf_prediction": np.where(rf_attack, "ATTACK", "BENIGN"),
            "anomaly_score": anomaly_scores,
            "is_anomaly": anomaly_prediction,
            "unified_decision": decision,
        }, index=traffic.index)
