import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection_engine.unified_detector import UnifiedDetector

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Network Traffic Analyzer",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "processed" / "binary_network_traffic.csv"
REPORT_FILE = BASE_DIR / "reports" / "traffic_report.csv"
MISSED_FILE = BASE_DIR / "models" / "missed_attacks.csv"
MODEL_FILE = BASE_DIR / "models" / "random_forest_binary.joblib"
FEATURE_IMPORTANCE_FILE = BASE_DIR / "models" / "feature_importance.csv"
ANOMALY_MODEL_FILE = BASE_DIR / "models" / "isolation_forest_v2.joblib"
ANOMALY_RESULTS_FILE = BASE_DIR / "reports" / "isolation_forest_v2_test_results.csv"


# ============================================================
# SAFE DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    report = pd.read_csv(REPORT_FILE, low_memory=False) if REPORT_FILE.exists() else pd.DataFrame()
    missed = pd.read_csv(MISSED_FILE, low_memory=False) if MISSED_FILE.exists() else pd.DataFrame()
    return df, report, missed


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_resource
def load_anomaly_bundle():
    if not ANOMALY_MODEL_FILE.exists():
        return None
    return joblib.load(ANOMALY_MODEL_FILE)


@st.cache_resource
def load_unified_detector():
    return UnifiedDetector(anomaly_percentile=95)


df, report, missed = load_data()
model = load_model()
anomaly_bundle = load_anomaly_bundle()

if "binary_label" not in df.columns:
    st.error("The dataset is missing the expected 'binary_label' column.")
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.title("🛡️ AI Network Traffic Analyzer")
st.markdown(
    """
    **AI-powered network traffic monitoring and intrusion detection dashboard**

    Analyze network traffic using a supervised Random Forest classifier and an
    unsupervised Isolation Forest anomaly detector.
    """
)
st.divider()

# ============================================================
# DATA SUMMARY
# ============================================================
total_records = len(df)
normal_count = int((pd.to_numeric(df["binary_label"], errors="coerce") == 0).sum())
attack_count = int((pd.to_numeric(df["binary_label"], errors="coerce") == 1).sum())
missed_count = len(missed)
attack_percentage = (attack_count / total_records * 100) if total_records else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{total_records:,}")
col2.metric("Normal Traffic", f"{normal_count:,}")
col3.metric("Attack Records", f"{attack_count:,}")
col4.metric("Missed Attacks", f"{missed_count:,}")
st.divider()

# ============================================================
# TABS
# ============================================================
tab_overview, tab_protocols, tab_attacks, tab_anomaly, tab_model, tab_data, tab_unified = st.tabs(
    [
        "📊 Overview",
        "🌐 Protocols",
        "🚨 Attack Analysis",
        "🔎 Anomaly Detection",
        "🤖 Models",
        "📁 Data & Export",
        "🤝 Unified Detection",
    ]
)

with tab_overview:
    st.subheader("Normal vs Attack Traffic")
    classification_df = pd.DataFrame({
        "Category": ["Normal", "Attack"],
        "Count": [normal_count, attack_count]
    })
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            classification_df, names="Category", values="Count", hole=0.45,
            title="Normal vs Malicious Traffic"
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "Label" in df.columns:
            attack_data = df[pd.to_numeric(df["binary_label"], errors="coerce") == 1]
            label_counts = attack_data["Label"].value_counts().head(15).rename_axis("Attack Type").reset_index(name="Count")
            fig = px.bar(label_counts, x="Count", y="Attack Type", orientation="h", title="Top Attack Types")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No multiclass 'Label' column was found for attack-type analysis.")

    st.subheader("Traffic Spike Analysis")
    if not report.empty and {"Elapsed Second", "Packet Count"}.issubset(report.columns):
        color_arg = "Status" if "Status" in report.columns else None
        fig = px.line(
            report, x="Elapsed Second", y="Packet Count", color=color_arg,
            markers=True, title="Packets Per Second"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Traffic spike chart is unavailable because reports/traffic_report.csv is missing or lacks the expected columns.")

    st.subheader("Missed Attack Analysis")
    if not missed.empty and "Label" in missed.columns:
        missed_counts = missed["Label"].value_counts().rename_axis("Attack Type").reset_index(name="Missed Count")
        fig = px.bar(missed_counts, x="Missed Count", y="Attack Type", orientation="h", title="Attacks Missed by Random Forest")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missed-attack records are available, or the report has no 'Label' column.")

with tab_protocols:
    st.subheader("Network Protocol Analysis")
    if "Protocol Type" in df.columns:
        protocol_map = {0: "HOPOPT", 1: "ICMP", 6: "TCP", 17: "UDP", 47: "GRE"}
        protocol_df = df.copy()
        protocol_df["Protocol"] = pd.to_numeric(protocol_df["Protocol Type"], errors="coerce").map(protocol_map).fillna("Unknown")
        protocol_counts = protocol_df["Protocol"].value_counts().rename_axis("Protocol").reset_index(name="Count")
        fig = px.bar(protocol_counts, x="Protocol", y="Count", text="Count", title="Network Protocol Distribution")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(xaxis_title="Protocol", yaxis_title="Number of Records", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if "binary_label" in protocol_df.columns:
            cross = pd.crosstab(protocol_df["Protocol"], protocol_df["binary_label"])
            cross = cross.rename(columns={0: "Benign", 1: "Attack"})
            st.markdown("**Protocol vs traffic label**")
            st.dataframe(cross, use_container_width=True)
    else:
        st.warning("The dataset does not contain a 'Protocol Type' column.")

with tab_attacks:
    st.subheader("Attack Type Distribution")
    if "Label" in df.columns:
        attack_data = df[pd.to_numeric(df["binary_label"], errors="coerce") == 1]
        label_counts = attack_data["Label"].value_counts().rename_axis("Attack Type").reset_index(name="Count")
        fig = px.bar(label_counts.head(25), x="Count", y="Attack Type", orientation="h", title="Attack Types (top 25)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'Label' column found.")

    st.subheader("Random Forest Missed Attacks")
    if not missed.empty:
        st.dataframe(missed, use_container_width=True)
        st.download_button(
            "Download missed attacks CSV",
            missed.to_csv(index=False).encode("utf-8"),
            file_name="missed_attacks.csv",
            mime="text/csv"
        )
    else:
        st.success("No missed-attack records are available in the saved report.")

with tab_anomaly:
    st.subheader("Isolation Forest Anomaly Detection")
    st.write(
        "Isolation Forest is an unsupervised detector. It flags records that look unusual; "
        "an anomaly is not automatically proof of a malicious attack."
    )

    if anomaly_bundle is None:
        st.warning(
            "Isolation Forest model not found. Expected file: "
            "`models/isolation_forest_v2.joblib`."
        )
        st.info(
            "Before using the detector, update `train_anomaly.py` to exclude all label columns "
            "(especially `binary_label`) from features, then retrain the model."
        )
    elif not isinstance(anomaly_bundle, dict) or "pipeline" not in anomaly_bundle or "feature_names" not in anomaly_bundle:
        st.error("The saved anomaly model has an unexpected format. Retrain it with the current train_anomaly.py script.")
    else:
        feature_names = anomaly_bundle["feature_names"]
        missing_features = [c for c in feature_names if c not in df.columns]
        if missing_features:
            st.error(f"The dataset is missing {len(missing_features)} feature(s) required by the model: {missing_features[:10]}")
        else:
            # Model was fitted using this exact feature list/order.
            X_anomaly = df[feature_names].copy()
            X_anomaly = X_anomaly.apply(pd.to_numeric, errors="coerce")
            X_anomaly = X_anomaly.replace([np.inf, -np.inf], np.nan)

            with st.spinner("Scoring dataset with Isolation Forest..."):
                try:
                    pipeline = anomaly_bundle["pipeline"]
                    # IsolationForest returns -1 for anomalies and 1 for inliers.
                    scores = -pipeline.decision_function(X_anomaly)

                    threshold = st.slider(
                        "Anomaly score threshold",
                        min_value=-0.05,
                        max_value=0.10,
                        value=0.0011,
                        step=0.0005,
                        key="anomaly_threshold",
                        help="Uses calibrated decision-function scores. Lower thresholds flag more records."
                    )

                    anomaly_results = df.copy()
                    anomaly_results["Anomaly Score"] = scores
                    anomaly_results["Anomaly Status"] = np.where(
                        scores >= threshold, "Anomaly", "Normal"
                    )

                    anomaly_count = int(
                        (anomaly_results["Anomaly Status"] == "Anomaly").sum()
                    )
                    anomaly_pct = (
                        anomaly_count / len(anomaly_results) * 100
                        if len(anomaly_results) else 0
                    )

                    k1, k2, k3 = st.columns(3)
                    k1.metric("Records Scored", f"{len(anomaly_results):,}")
                    k2.metric("Flagged Anomalies", f"{anomaly_count:,}")
                    k3.metric("Anomaly Rate", f"{anomaly_pct:.2f}%")

                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        status_counts = anomaly_results["Anomaly Status"].value_counts().rename_axis("Status").reset_index(name="Count")
                        fig = px.pie(status_counts, names="Status", values="Count", hole=0.45, title="Normal vs Anomaly")
                        st.plotly_chart(fig, use_container_width=True)
                    with chart_col2:
                        fig = px.histogram(
                            anomaly_results, x="Anomaly Score", color="Anomaly Status",
                            nbins=50, barmode="overlay", opacity=0.7,
                            title="Anomaly Score Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("**Most unusual records**")
                    st.caption("Sorted by anomaly score, highest first. Scores are relative model scores, not calibrated probabilities.")
                    flagged = anomaly_results[anomaly_results["Anomaly Status"] == "Anomaly"].sort_values("Anomaly Score", ascending=False)
                    display_cols = [c for c in ["Anomaly Status", "Anomaly Score", "Label", "binary_label", "Protocol Type", "Source IP", "Destination IP"] if c in flagged.columns]
                    if not display_cols:
                        display_cols = ["Anomaly Status", "Anomaly Score"]
                    st.dataframe(flagged[display_cols].head(500), use_container_width=True)

                    st.download_button(
                        "Download all anomaly results (CSV)",
                        anomaly_results.to_csv(index=False).encode("utf-8"),
                        file_name="isolation_forest_anomaly_results.csv",
                        mime="text/csv"
                    )

                    # Optional ground-truth evaluation for labelled datasets.
                    if "binary_label" in anomaly_results.columns:
                        st.divider()
                        st.markdown("**Label comparison (exploratory, full dataset)**")
                        st.caption(
                            "This compares model flags with existing labels for inspection. "
                            "For formal performance metrics, use the held-out test report, not this full-dataset view."
                        )
                        true_binary = pd.to_numeric(anomaly_results["binary_label"], errors="coerce")
                        valid_mask = true_binary.isin([0, 1])
                        if valid_mask.any():
                            compare = pd.crosstab(
                                true_binary[valid_mask].map({0: "BENIGN", 1: "ATTACK"}),
                                anomaly_results.loc[valid_mask, "Anomaly Status"],
                                rownames=["True label"], colnames=["Detector output"]
                            )
                            st.dataframe(compare, use_container_width=True)

                except Exception as exc:
                    st.error(f"Anomaly scoring failed: {exc}")
                    st.info("Check that the model was retrained with the current feature list and that all required columns exist.")

    if ANOMALY_RESULTS_FILE.exists():
        st.markdown("**Saved test-set evaluation**")
        try:
            eval_df = pd.read_csv(ANOMALY_RESULTS_FILE, low_memory=False)
            if {"true_label", "predicted_label"}.issubset(eval_df.columns):
                st.dataframe(eval_df.head(100), use_container_width=True)
                st.download_button(
                    "Download saved test-set predictions",
                    eval_df.to_csv(index=False).encode("utf-8"),
                    file_name="isolation_forest_test_results.csv",
                    mime="text/csv",
                    key="download_saved_anomaly_test"
                )
            else:
                st.caption("The saved evaluation CSV does not contain the expected true_label and predicted_label columns.")
        except Exception as exc:
            st.warning(f"Could not read saved anomaly evaluation file: {exc}")

with tab_model:
    st.subheader("Machine Learning Models")
    col1, col2, col3 = st.columns(3)
    col1.write("**Supervised model**")
    col1.write("Random Forest Classifier")
    col2.write("**Random Forest input features**")
    col2.write(getattr(model, "n_features_in_", "Not available"))
    col3.write("**Random Forest classes**")
    classes = getattr(model, "classes_", [])
    col3.write(", ".join(str(x) for x in classes) if len(classes) else "Not available")

    st.divider()
    st.write("**Isolation Forest**")
    if anomaly_bundle and isinstance(anomaly_bundle, dict):
        st.success("Saved Isolation Forest bundle found.")
        st.write(f"Features used: {len(anomaly_bundle.get('feature_names', []))}")
        st.write(f"Model file: `{ANOMALY_MODEL_FILE.relative_to(BASE_DIR)}`")
    else:
        st.warning("Isolation Forest model bundle is not available.")

    if FEATURE_IMPORTANCE_FILE.exists():
        feature_df = pd.read_csv(FEATURE_IMPORTANCE_FILE)
        st.subheader("Feature Importance")
        st.dataframe(feature_df, use_container_width=True)
        if {"feature", "importance"}.issubset(feature_df.columns):
            feature_chart_df = feature_df.sort_values("importance", ascending=False).head(15)
            fig = px.bar(feature_chart_df, x="importance", y="feature", orientation="h", title="Top 15 Important Features")
            st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.subheader("Dataset Information")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Total Features", f"{df.shape[1]:,}")
    info_col2.metric("Normal Records", f"{normal_count:,}")
    info_col3.metric("Attack Percentage", f"{attack_percentage:.2f}%")

    st.write("**Dataset preview**")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Export Data")
    st.download_button(
        "Download Network Dataset",
        df.to_csv(index=False).encode("utf-8"),
        file_name="network_traffic_analysis.csv",
        mime="text/csv"
    )
    if not report.empty:
        st.download_button(
            "Download Traffic Report",
            report.to_csv(index=False).encode("utf-8"),
            file_name="traffic_report.csv",
            mime="text/csv"
        )
    if not missed.empty:
        st.download_button(
            "Download Missed Attacks",
            missed.to_csv(index=False).encode("utf-8"),
            file_name="missed_attacks.csv",
            mime="text/csv"
        )


with tab_unified:
    st.subheader("Unified Random Forest + Isolation Forest v2")
    st.write(
        "Compare supervised attack predictions with unsupervised anomaly scores. "
        "Disagreements remain visible for investigation."
    )

    try:
        unified_detector = load_unified_detector()

        threshold_col, sample_col = st.columns(2)
        with threshold_col:
            unified_threshold = st.slider(
                "Anomaly score threshold",
                min_value=-0.05,
                max_value=0.10,
                value=0.001112,
                step=0.0005,
                format="%.5f",
                key="unified_anomaly_threshold",
                help="Higher scores indicate more unusual traffic. This is not a probability.",
            )
        with sample_col:
            unified_sample_size = st.slider(
                "Records to analyze",
                min_value=1000,
                max_value=min(10000, max(1000, len(df))),
                value=min(10000, max(1000, len(df))),
                step=1000,
                key="unified_sample_size",
            )

        st.caption(
            "The initial threshold is approximately the previously evaluated "
            "95th-percentile setting. Adjust it to explore the trade-off between "
            "false alarms and missed anomalies."
        )

        if st.button("Run Unified Detection", type="primary", key="run_unified_detection"):
            # Use a reproducible sample so the dashboard remains responsive.
            sample_count = min(int(unified_sample_size), len(df))
            traffic_sample = (
                df.sample(n=sample_count, random_state=42)
                if sample_count < len(df)
                else df.copy()
            )

            unified_detector.anomaly_threshold = float(unified_threshold)
            predictions = unified_detector.predict(traffic_sample)

            results = predictions.reset_index(drop=True).copy()
            results.insert(0, "dataset_index", traffic_sample.index.to_numpy())

            if "Label" in traffic_sample.columns:
                results["actual_label"] = traffic_sample["Label"].astype(str).to_numpy()
            if "binary_label" in traffic_sample.columns:
                results["actual_binary_label"] = traffic_sample["binary_label"].to_numpy()

            st.session_state["unified_detection_results"] = results
            st.session_state["unified_detection_threshold_used"] = float(unified_threshold)

        results = st.session_state.get("unified_detection_results")

        if results is None:
            st.info("Choose your threshold and sample size, then click Run Unified Detection.")
        else:
            used_threshold = st.session_state.get("unified_detection_threshold_used")
            st.caption(f"Threshold used for these results: {used_threshold:.5f}")

            total = len(results)
            both = int((results["unified_decision"] == "ATTACK_CONFIRMED_BY_BOTH").sum())
            rf_only = int((results["unified_decision"] == "ATTACK_RF_ONLY").sum())
            anomaly_only = int((results["unified_decision"] == "SUSPICIOUS_ANOMALY_ONLY").sum())
            benign_candidate = int((results["unified_decision"] == "BENIGN_CANDIDATE").sum())

            metric_cols = st.columns(4)
            metric_cols[0].metric("Flagged by both", f"{both:,}")
            metric_cols[1].metric("Random Forest only", f"{rf_only:,}")
            metric_cols[2].metric("Anomaly only", f"{anomaly_only:,}")
            metric_cols[3].metric("Neither flagged", f"{benign_candidate:,}")

            st.subheader("Decision breakdown")
            summary = (
                results["unified_decision"]
                .value_counts()
                .rename_axis("Unified decision")
                .reset_index(name="Records")
            )
            fig = px.bar(
                summary,
                x="Unified decision",
                y="Records",
                title="Records by unified decision",
                text="Records",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True, height=420)

            st.download_button(
                "Download Unified Detection CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="unified_detection_results.csv",
                mime="text/csv",
                key="download_unified_detection_csv",
            )

            st.warning(
                "These are model-generated indicators, not proof of malicious activity. "
                "Review flagged traffic and investigate disagreements before taking action."
            )

    except Exception as exc:
        st.error(f"Unified Detection could not run: {exc}")
        st.info(
            "Check that both saved models exist, their feature names and order match, "
            "and the dataset contains all required model features."
        )

st.divider()
st.caption("AI Network Traffic Analyzer • Random Forest + Isolation Forest • Dashboard V2")
