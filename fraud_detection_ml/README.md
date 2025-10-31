# Fraud Detection ML

A fast, reproducible credit card fraud detection pipeline optimized for **recall on fraud class (class=1)**.

## Overview

This project addresses extreme class imbalance in credit card fraud detection. The pipeline implements two approaches:
- **Baseline**: IsolationForest (unsupervised anomaly detection)
- **Champion**: SMOTE + XGBoost (supervised learning with oversampling)

## Quickstart

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate    # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py --csv creditcard.csv
```

## Methods

### Baseline: IsolationForest
- Unsupervised anomaly detection
- Contamination="auto", 200 estimators
- Maps predictions: -1→fraud(1), 1→non-fraud(0)

### Champion: SMOTE + XGBoost
- SMOTE oversampling on training data only
- XGBoost with optimized hyperparameters
- Binary classification with 0.5 threshold

## Results

*Results will be auto-filled after first run from outputs/metrics.json*

**Fraud Recall (class=1):**
- IsolationForest: [TBD]
- SMOTE+XGBoost: [TBD]

**Precision (class=1):**
- IsolationForest: [TBD] 
- SMOTE+XGBoost: [TBD]

**Confusion Matrix Images:**
- `outputs/confusion_matrix_IsolationForest.png`
- `outputs/confusion_matrix_SMOTE_XGBoost.png`

## Output Artifacts

The pipeline creates `./outputs/` with:
- `metrics.json` - Combined metrics for both models
- `classification_report.txt` - Detailed classification reports
- `confusion_matrix_*.png` - Confusion matrix visualizations
- `run.log` - Execution log with timings

## Notes

- **Threshold Tuning**: Consider adjusting threshold (0.35-0.45) for higher recall
- **Alternative**: Remove SMOTE and set `scale_pos_weight ≈ (neg/pos)` in XGBoost
- **Reproducible**: All random seeds fixed for deterministic results
- **Fast**: Single CSV input, no external dependencies

## CLI Arguments

```bash
python main.py --csv creditcard.csv --seed 42 --test_size 0.2
```

- `--csv`: Path to credit card CSV file (default: "creditcard.csv")
- `--seed`: Random seed for reproducibility (default: 42)
- `--test_size`: Test set proportion (default: 0.2)
