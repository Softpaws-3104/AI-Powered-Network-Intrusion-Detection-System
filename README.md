# AI Network Traffic Analyzer

A Python-based cybersecurity project that analyzes network traffic, classifies potential attacks, detects unusual activity, and presents results through an interactive Streamlit dashboard.

## Features

- **Network traffic analysis:** Analyze packet-capture data and identify protocols, source IPs, and destination IPs.
- **Random Forest classifier:** Classify network traffic as benign or potentially malicious.
- **Isolation Forest anomaly detection:** Identify traffic patterns that differ from normal behavior.
- **Unified detection engine:** Combine the outputs of both models to distinguish confirmed alerts, classifier-only alerts, anomaly-only alerts, and benign candidates.
- **Interactive dashboard:** Explore traffic statistics, attack analysis, anomaly results, and model comparisons.
- **Evaluation reports:** Review model metrics and threshold comparisons.
- **CSV export:** Download detection results for further analysis.

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
network-traffic-analyzer/
├── analyzer.py
├── prepare_dataset.py
├── train_model.py
├── train_anomaly.py
├── dashboard/
│   └── app.py
├── detection_engine/
│   ├── unified_detector.py
│   ├── compare_models.py
│   └── evaluation scripts
├── models/
│   ├── random_forest_binary.joblib
│   ├── isolation_forest.joblib
│   └── isolation_forest_v2.joblib
└── reports/
    ├── fair_model_comparison.csv
    ├── unified_holdout_metrics.csv
    └── other evaluation summaries
```

## Model Evaluation

In the current evaluation results:

| Model / strategy | Accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Random Forest | 99.15% | 99.96% | 98.33% | 99.14% |
| Isolation Forest | 84.99% | 92.77% | 75.89% | 83.49% |
| Unified: either model flags | 96.42% | 94.35% | 98.75% | 96.50% |

These results depend on the dataset, decision thresholds, and evaluation setup. The unified strategy prioritizes attack recall but can generate more false alarms. The figures are experimental results, not a guarantee of real-world detection performance.

## Running the Dashboard

1. Install Python and create a virtual environment.
2. Install the project's required Python packages.
3. Ensure the trained model files and processed dataset are available in the expected paths.
4. From the project root, run:

```bash
streamlit run dashboard/app.py
```

## Responsible Use

Use this project only on network traffic you own or are authorized to analyze. Detection results are indicators for investigation, not definitive proof of malicious activity.

## Author

Developed as a cybersecurity and machine-learning project to explore network traffic classification, anomaly detection, and interactive security monitoring.
