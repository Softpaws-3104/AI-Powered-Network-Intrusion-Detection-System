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
from live_monitor.live_feature_extractor import capture_batches
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI-Powered Network Intrusion Detection and Anomaly Analysis System",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# GLOBAL VISUAL THEME — consistent with the project header
# ============================================================
px.defaults.template = "plotly_dark"
st.markdown(
    """
    <style>
    :root {
        --page-bg: #0b1119;
        --panel-bg: #101b24;
        --panel-bg-2: #12232b;
        --teal: #2dd4bf;
        --cyan: #7dd3fc;
        --main-text: #e7eef5;
        --muted-text: #9aaabd;
        --panel-border: rgba(45, 212, 191, 0.22);
    }
    .stApp {
        background: radial-gradient(ellipse at 5% 0%, #10272b 0%, #0b1119 43%, #0b1119 100%);
        color: var(--main-text);
    }
    [data-testid="stHeader"] { background: rgba(11, 17, 25, 0.88); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1600px; }
    h1, h2, h3, h4 { color: #edf4fa !important; letter-spacing: -0.02em; }
    h2, h3 { margin-top: 0.45rem; }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: #c7d3df; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(18, 35, 43, 0.98), rgba(13, 23, 32, 0.98));
        border: 1px solid var(--panel-border);
        border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        min-height: 112px;
    }
    [data-testid="stMetricLabel"] { color: #9fb3c5 !important; font-size: 0.88rem; }
    [data-testid="stMetricValue"] { color: #eff8ff !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { color: var(--teal) !important; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 7px; border-bottom: 1px solid rgba(45, 212, 191, 0.18);
        background: rgba(12, 22, 30, 0.78); padding: 8px; border-radius: 13px 13px 0 0;
    }
    [data-testid="stTabs"] button[role="tab"] {
        color: #a9bac9; border-radius: 9px; padding: 10px 13px;
        border: 1px solid transparent; transition: all 120ms ease-in-out;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        color: #e8ffff; background: rgba(45, 212, 191, 0.08);
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #5eead4; background: rgba(45, 212, 191, 0.10);
        border: 1px solid rgba(45, 212, 191, 0.26);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(45, 212, 191, 0.20) !important; border-radius: 14px !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid rgba(45, 212, 191, 0.18); border-radius: 12px; overflow: hidden;
    }
    [data-testid="stDataFrame"] [role="grid"] { background: #101a23; }
    [data-testid="stPlotlyChart"] {
        background: linear-gradient(145deg, rgba(16, 27, 36, 0.78), rgba(12, 20, 29, 0.72));
        border: 1px solid rgba(45, 212, 191, 0.16); border-radius: 14px;
        padding: 8px 8px 0 8px;
    }
    .stButton > button, .stDownloadButton > button {
        color: #eafffb; background: linear-gradient(110deg, #0f766e, #115e59);
        border: 1px solid rgba(94, 234, 212, 0.42); border-radius: 10px;
        padding: 0.55rem 1rem; font-weight: 650;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        color: #ffffff; border-color: #5eead4; background: linear-gradient(110deg, #0d9488, #0f766e);
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #2dd4bf; border-color: #2dd4bf;
    }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        background: #101b24; border-color: rgba(125, 211, 252, 0.24); border-radius: 9px;
    }
    [data-testid="stAlert"] { border-radius: 12px; border: 1px solid rgba(125, 211, 252, 0.20); }
    hr { border-color: rgba(125, 211, 252, 0.18) !important; }
    code { color: #7dd3fc; background: rgba(125, 211, 252, 0.08); }
    @media (max-width: 800px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        [data-testid="stMetric"] { padding: 13px; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 3px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
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
# OFFICIAL PROJECT HEADER
# ============================================================
st.markdown(
    """
    <div style="
        background: linear-gradient(110deg, rgba(13, 35, 40, 0.98), rgba(12, 18, 28, 0.99));
        border: 1px solid rgba(45, 212, 191, 0.28);
        border-radius: 16px;
        padding: 28px 30px;
        margin-bottom: 22px;
    ">
        <div style="color: #2dd4bf; font-size: 0.76rem; font-weight: 700;
                    letter-spacing: 2px; margin-bottom: 12px;">
            NETWORK SECURITY / MACHINE LEARNING
        </div>
        <h1 style="margin: 0; font-size: 2.0rem; line-height: 1.3; color: #f1f5f9;">
            AI-Powered Network Intrusion Detection and Anomaly Analysis System
        </h1>
        <div style="margin-top: 9px; font-size: 1.08rem; color: #7dd3fc;">
            Network Traffic Classification, Anomaly Analysis, and Investigation
        </div>
        <p style="margin: 20px 0 0 0; max-width: 850px; line-height: 1.75; color: #cbd5e1;">
            A machine-learning platform for classifying network traffic, identifying
            anomalous behavior, and comparing detection outcomes for security investigation.
        </p>
        <div style="margin-top: 22px; padding-top: 14px;
                    border-top: 1px solid rgba(148, 163, 184, 0.2);
                    color: #94a3b8; font-size: 0.82rem;">
            OFFLINE DATASET ANALYSIS &nbsp; &#8226; &nbsp; LIVE DETECTION UNDER DEVELOPMENT
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Machine-learning dashboard for prepared network traffic datasets")
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
tab_overview, tab_protocols, tab_attacks, tab_anomaly, tab_model, tab_data, tab_unified, tab_live = st.tabs(
    [
        "📊 Overview",
        "🌐 Protocols",
        "🚨 Attack Analysis",
        "🔎 Anomaly Detection",
        "🤖 Models",
        "📁 Data & Export",
        "🤝 Unified Detection",
        "📡 Live Detection",
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



with tab_live:
    st.subheader("Live Network Detection")
    st.write(
        "Capture a bounded sample of network traffic, aggregate compatible packets "
        "into 10-packet feature batches, and inspect predictions from the existing models."
    )
    st.warning(
        "Live predictions are experimental. They are not proof of malicious activity, "
        "and live feature distributions have not yet been validated against the training data."
    )

    interface = st.selectbox(
        "Network interface",
        options=["eth0", "wlan0"],
        index=0,
        key="live_interface",
        help="Choose a specific network interface that exists on this system.",
    )
    packet_limit = st.select_slider(
        "Maximum packets to capture",
        options=[10, 20, 30, 50, 100],
        value=30,
        key="live_packet_limit",
    )
    capture_timeout = st.slider(
        "Capture timeout (seconds)",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
        key="live_capture_timeout",
    )
    live_threshold = st.number_input(
        "Anomaly score threshold",
        min_value=-0.05,
        max_value=0.10,
        value=0.001112,
        step=0.0005,
        format="%.6f",
        key="live_anomaly_threshold",
        help="Higher scores indicate more unusual traffic. This is not a probability.",
    )

    if st.button("Capture and Analyze Live Traffic", type="primary", key="run_live_detection"):
        # Clear previous results so failed captures cannot display stale detections.
        st.session_state.pop("live_detection_results", None)
        st.session_state.pop("live_detection_threshold", None)
        st.session_state.pop("live_capture_stats", None)

        try:
            with st.spinner("Capturing packets and building feature batches..."):
                live_features = capture_batches(
                    iface=interface,
                    count=int(packet_limit),
                    timeout=int(capture_timeout),
                )

            st.session_state["live_capture_stats"] = {
                "packets_requested": int(packet_limit),
                "compatible_rows": len(live_features) * 10,
                "complete_batches": len(live_features),
                "features": len(live_features.columns),
                "interface": interface,
            }

            if live_features.empty:
                st.warning(
                    "No complete 10-packet feature batches were produced. "
                    "Try a longer timeout or generate ordinary network traffic, then retry."
                )
                st.session_state.pop("live_detection_results", None)
            else:
                detector = UnifiedDetector()
                detector.anomaly_threshold = float(live_threshold)
                live_predictions = detector.predict(live_features)
                live_results = live_predictions.reset_index(drop=True).copy()
                live_results.insert(0, "batch_number", range(1, len(live_results) + 1))
                st.session_state["live_detection_results"] = live_results
                st.session_state["live_detection_threshold"] = float(live_threshold)

        except PermissionError:
            st.error(
                "Packet capture permission was denied. Check dumpcap permissions and "
                "your system's authorized packet-capture configuration. Avoid running the "
                "entire dashboard as root."
            )
        except Exception as exc:
            st.error(f"Live capture or inference failed: {exc}")

    stats = st.session_state.get("live_capture_stats")
    if stats:
        st.caption(f"Interface: {stats['interface']}")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Complete batches", stats["complete_batches"])
        metric_cols[1].metric("Features per batch", stats["features"])
        metric_cols[2].metric("Compatible rows in complete batches", stats["compatible_rows"])

    live_results = st.session_state.get("live_detection_results")
    if live_results is not None:
        st.subheader("Live Detection Results")
        st.caption(
            f"Anomaly threshold used: "
            f"{st.session_state.get('live_detection_threshold', 0.001112):.6f}"
        )
        st.dataframe(live_results, use_container_width=True)
        st.download_button(
            "Download Live Detection CSV",
            data=live_results.to_csv(index=False).encode("utf-8"),
            file_name="live_detection_results.csv",
            mime="text/csv",
            key="download_live_detection_csv",
        )

st.divider()
st.caption("AI-Powered Network Intrusion Detection and Anomaly Analysis System | Random Forest + Isolation Forest | Dashboard V2")
