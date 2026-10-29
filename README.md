# AI-Powered Network Intrusion Detection and Anomaly Analysis System

An AI/ML-based cybersecurity project that analyzes network traffic, classifies potentially malicious activity, detects anomalous patterns, and presents security insights through an interactive Streamlit dashboard.

## Features

- **Network traffic classification:** Random Forest model to distinguish benign traffic from potentially malicious traffic.
- **Anomaly detection:** Isolation Forest model to identify unusual network behavior.
- **Unified detection engine:** Combines classification and anomaly signals for comprehensive traffic analysis.
- **Live traffic feature extraction:** Extracts and aggregates network packet features for live analysis.
- **Interactive dashboard:** Displays traffic statistics, detection results, anomaly scores, and model evaluation.
- **Model evaluation:** Examines detection performance, thresholds, false positives, missed attacks, and duplicate-aware evaluation.
- **Reports:** Stores evaluation summaries and supports CSV-based analysis.

## Technology Stack

### Programming Languages
- **Python 3.13** — core application, machine learning, packet processing, and dashboard backend.
- **SQL / SQLite** — used in a temporary dataset-audit workflow.
- **HTML and CSS** — dashboard markup and styling embedded in the Streamlit application.
- **Bash / Zsh** — Linux commands, environment management, and project workflows.

### Python Libraries
- **pandas and NumPy** — dataset preparation and numerical processing.
- **scikit-learn** — Random Forest and Isolation Forest models and evaluation.
- **Scapy** — packet capture and packet-level feature extraction.
- **Streamlit** — interactive web dashboard.
- **Plotly** — charts and visualizations.
- **joblib** — saving and loading trained models.

### Security and Development Tools
- **Kali Linux** — cybersecurity development environment.
- **tcpdump and Wireshark** — packet capture and traffic inspection.
- **Git and GitHub** — version control and remote repository backup.

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
├── live_monitor/
│   └── live_feature_extractor.py
├── models/
│   ├── isolation_forest.joblib
│   ├── isolation_forest_v2.joblib
│   ├── random_forest_binary.joblib
│   └── random_forest_without_number_DIAGNOSTIC.joblib
├── reports/
│   └── duplicate_aware_evaluation.txt
├── prepare_dataset.py
├── train_anomaly.py
├── train_model.py
└── README.md
```

The structure above highlights the main project components; additional files may exist in the repository.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the core dependencies:

```bash
pip install pandas numpy scikit-learn scapy streamlit plotly joblib
```

Install any additional dependencies required by individual project scripts.

## Run the Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

Alternatively, if using the project's activated virtual environment:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

## Model Training

The repository includes scripts for training the classification and anomaly-detection models:

```bash
python3 train_model.py
python3 train_anomaly.py
```

Training requires the appropriate prepared dataset and dependencies. Review the scripts and dataset paths before running them. Retraining may replace existing model artifacts, so back up important models first.

## Evaluation and Limitations

Model evaluation has been performed on the available processed dataset. Duplicate-aware evaluation is also used to investigate whether repeated feature patterns affect results.

- High offline evaluation scores do not guarantee equivalent performance on unseen networks or live traffic.
- The source provenance and labeling of the dataset must be verified before making strong claims about generalization.
- Live feature extraction must match the feature definitions and aggregation process used during training.
- Unusual traffic or an anomaly flag does not, by itself, prove that an attack occurred.
- The diagnostic Random Forest model without the `Number` feature is for comparison; it does not replace the production model.

## Data and Model Notes

- Raw datasets, processed datasets, packet captures, and local backups are excluded from Git by default.
- Trained model files are stored in `models/`.
- Evaluation summaries are stored in `reports/`.
- Model performance depends on dataset quality, feature preparation, threshold selection, and evaluation methodology.

## Live Traffic Monitoring

The live monitoring component captures authorized network traffic and converts accepted IPv4/ARP packets into feature batches for inference.

- Tested interface: `eth0` in the current Kali environment.
- Tested capture limit: up to 100 packets with a bounded timeout.
- Feature aggregation: batches of 10 accepted packets.
- Feature schema: 39 features in the expected model input order.
- Incomplete batches are discarded.
- The unified detector returns Random Forest classification results, Isolation Forest anomaly scores, and a combined decision.

Run the dashboard from the project root:

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

Live packet capture should only be used on networks and interfaces you are authorized to monitor.

## Evaluation Results

### Random Forest

The existing group-aware diagnostic evaluation reported:

| Metric | Result |
|---|---:|
| Accuracy | 99.15% |
| Precision | 99.95% |
| Attack recall | 98.34% |
| F1 score | 99.14% |
| ROC-AUC | 0.9977 |

### Isolation Forest

The existing evaluation reported:

| Metric | Result |
|---|---:|
| Accuracy | 84.63% |
| Attack precision | 92.83% |
| Attack recall | 75.06% |
| Attack F1 score | 83.00% |
| ROC-AUC | 0.9455 |

These are offline results from the available dataset. They do not establish equivalent detection performance on real-world network traffic.

## Production and Diagnostic Models

The dashboard currently uses these production model paths:

- `models/random_forest_binary.joblib` — Random Forest classifier.
- `models/isolation_forest_v2.joblib` — Isolation Forest model bundle.

Other model files are diagnostic or experimental artifacts. Do not treat them as production replacements without separate evaluation and an explicit model-selection decision.

## Additional Limitations and Safe Training Practices

- The dataset's `Number` feature has a substantial association with the class distribution and can influence individual predictions.
- Live feature extraction and inference have passed structural tests, but complete equivalence between live aggregation and training-time feature generation has not been established.
- An anomaly flag indicates unusual behavior; it does not prove that an attack occurred.
- Live predictions have not been validated against a sufficiently representative set of labeled real-world traffic.
- Inspect training-script input and output paths before running them. In particular, verify `train_model.py` because its previously inspected base-directory setting points to a different project directory.
- Back up existing model artifacts before retraining. Do not replace production models with diagnostic or experimental models without reviewing their evaluation results.

## Project Goal

Develop a practical machine-learning-based network security tool that combines intrusion classification, anomaly detection, and interactive analysis to support cybersecurity investigation.

## Disclaimer

Use network monitoring and packet capture only on systems and networks you own or are explicitly authorized to analyze.
