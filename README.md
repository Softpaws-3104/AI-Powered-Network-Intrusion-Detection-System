# AI-Powered Network Intrusion Detection and Anomaly Analysis System

An AI/ML-based cybersecurity project that analyzes network traffic,
classifies potentially malicious activity, detects anomalous patterns,
and presents security insights through an interactive Streamlit dashboard.

## Features

- **Network traffic classification:** Use a Random Forest model to
  distinguish benign traffic from potentially malicious traffic.
- **Anomaly detection:** Use Isolation Forest to identify unusual
  network behavior.
- **Unified detection engine:** Combine classification and anomaly
  signals to support more comprehensive traffic analysis.
- **Interactive dashboard:** Explore traffic statistics, detection
  results, anomaly scores, and model evaluation.
- **Model evaluation:** Compare detection performance, decision
  thresholds, false positives, and missed attacks.
- **CSV reports:** Export or inspect analysis results for further review.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Scapy
- Streamlit
- tcpdump and Wireshark
- joblib

## Project Structure

```text
AI-Powered-Network-Intrusion-Detection-System/
├── dashboard/
│   └── app.py
├── detection_engine/
│   ├── compare_models.py
│   ├── evaluate_saved_thresholds.py
│   ├── evaluate_unified_holdout.py
│   ├── test_unified_detector.py
│   ├── tune_anomaly_threshold.py
│   ├── unified_detector.py
│   └── validate_saved_anomaly.py
├── models/
│   ├── isolation_forest.joblib
│   ├── isolation_forest_v2.joblib
│   └── random_forest_binary.joblib
├── reports/
├── prepare_dataset.py
├── train_anomaly.py
├── train_model.py
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn scapy streamlit joblib
```

Install any additional dependencies required by the project scripts.

## Run the Dashboard

From the project root, run:

```bash
streamlit run dashboard/app.py
```

## Model Training

The repository includes training scripts for the classification
and anomaly-detection models.

```bash
python3 train_model.py
python3 train_anomaly.py
```

Training requires the appropriate prepared dataset and dependencies.
Review the scripts and dataset paths before running them.

## Data and Model Notes

- Raw datasets, processed datasets, packet captures, and local backups
  are excluded from Git by default.
- Trained model files are stored in `models/`.
- Evaluation summaries are stored in `reports/`.
- Model performance depends on the dataset, feature preparation,
  threshold selection, and evaluation methodology.
- Offline dataset performance does not guarantee equivalent
  performance on live network traffic.

## Project Goal

Develop a practical machine-learning-based network security tool
that combines intrusion classification, anomaly detection, and
interactive analysis to support cybersecurity investigation.

## Disclaimer

Use network monitoring and packet capture only on systems and networks
you own or are explicitly authorized to analyze.
