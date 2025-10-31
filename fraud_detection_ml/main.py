#!/usr/bin/env python3
"""
Fraud Detection ML Pipeline
Implements IsolationForest baseline and SMOTE + XGBoost champion model.
Optimizes for recall on fraud class (class=1).
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def ensure_dirs():
    """Create outputs directory if it doesn't exist."""
    Path("outputs").mkdir(exist_ok=True)


def log(message, level=logging.INFO):
    """Log message to both console and file."""
    print(message)
    logging.info(message)


def load_data(csv_path):
    """Load and return the credit card dataset."""
    log(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    log(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df


def summarize_imbalance(df):
    """Print class imbalance summary."""
    fraud_count = df['Class'].sum()
    non_fraud_count = len(df) - fraud_count
    fraud_ratio = fraud_count / len(df)

    log(f"Class distribution:")
    log(f"  Non-fraud: {non_fraud_count:,} ({1-fraud_ratio:.4f})")
    log(f"  Fraud: {fraud_count:,} ({fraud_ratio:.4f})")
    log(f"  Fraud ratio: {fraud_ratio:.4f}")

    return {
        "non_fraud_count": int(non_fraud_count),
        "fraud_count": int(fraud_count),
        "fraud_ratio": float(fraud_ratio)
    }


def preprocess(df):
    """Preprocess data: drop Time, scale Amount, keep V1-V28 unchanged."""
    log("Preprocessing data...")

    # Drop Time column
    df_processed = df.drop('Time', axis=1)

    # Scale only Amount column
    scaler = StandardScaler()
    df_processed['Amount'] = scaler.fit_transform(df_processed[['Amount']])

    log("Preprocessing complete: dropped Time, scaled Amount")
    return df_processed


def split(df, test_size, random_state):
    """Split data into train/test sets."""
    log(f"Splitting data: {1-test_size:.1f}/{test_size:.1f} train/test")

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    log(f"Train set: {len(X_train)} samples")
    log(f"Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def isolation_forest_baseline(X_train, X_test, y_test):
    """Train and evaluate IsolationForest baseline."""
    log("Training IsolationForest baseline...")

    # Train IsolationForest
    iso_forest = IsolationForest(
        contamination="auto",
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    )
    iso_forest.fit(X_train)

    # Predict and map: -1->fraud(1), 1->non-fraud(0)
    predictions = iso_forest.predict(X_test)
    predictions_mapped = np.where(predictions == -1, 1, 0)

    log("IsolationForest training complete")
    return predictions_mapped


def smote_xgb(X_train, y_train, X_test):
    """Train SMOTE + XGBoost model."""
    log("Training SMOTE + XGBoost...")

    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    log(f"SMOTE applied: {len(X_train_smote)} samples after resampling")

    # Train XGBoost
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_depth=6,
        learning_rate=0.08,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=1.0,  # SMOTE handles imbalance
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train_smote, y_train_smote)

    # Predict probabilities and apply threshold
    y_pred_proba = xgb.predict_proba(X_test)[:, 1]
    predictions = (y_pred_proba > 0.5).astype(int)

    log("SMOTE + XGBoost training complete")
    return predictions


def eval_and_save(y_true, y_pred, model_name, output_dir):
    """Evaluate model and save artifacts."""
    log(f"Evaluating {model_name}...")

    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()

    classes = ['Non-Fraud', 'Fraud']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_path = f"{output_dir}/confusion_matrix_{model_name}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    log(f"Saved confusion matrix: {cm_path}")

    # Extract per-class metrics as simple floats
    class_metrics = {k: v for k, v in report.items() if k in ('0', '1')}
    precision = {cls: float(vals.get('precision', 0.0)) for cls, vals in class_metrics.items()}
    recall = {cls: float(vals.get('recall', 0.0)) for cls, vals in class_metrics.items()}
    f1 = {cls: float(vals.get('f1-score', 0.0)) for cls, vals in class_metrics.items()}

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path
    }


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Fraud Detection ML Pipeline")
    parser.add_argument("--csv", default="creditcard.csv", help="Path to CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")

    args = parser.parse_args()

    # Setup logging
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/run.log'),
            logging.StreamHandler()
        ]
    )

    start_time = time.time()
    log("Starting Fraud Detection ML Pipeline")
    log(f"Arguments: csv={args.csv}, seed={args.seed}, test_size={args.test_size}")

    try:
        # Load and preprocess data
        df = load_data(args.csv)
        imbalance_stats = summarize_imbalance(df)
        df_processed = preprocess(df)

        # Split data
        X_train, X_test, y_train, y_test = split(df_processed, args.test_size, args.seed)

        # Train and evaluate models
        iso_pred = isolation_forest_baseline(X_train, X_test, y_test)
        xgb_pred = smote_xgb(X_train, y_train, X_test)

        # Evaluate models
        iso_metrics = eval_and_save(y_test, iso_pred, "IsolationForest", "outputs")
        xgb_metrics = eval_and_save(y_test, xgb_pred, "SMOTE_XGBoost", "outputs")

        # Save combined metrics
        runtime = time.time() - start_time
        metrics = {
            "data": imbalance_stats,
            "IsolationForest": iso_metrics,
            "SMOTE_XGBoost": xgb_metrics,
            "runtime_seconds": round(runtime, 2)
        }

        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save classification reports
        with open("outputs/classification_report.txt", "w") as f:
            f.write("=== IsolationForest Results ===\n")
            f.write(classification_report(y_test, iso_pred))
            f.write("\n\n=== SMOTE + XGBoost Results ===\n")
            f.write(classification_report(y_test, xgb_pred))

        log(f"Pipeline completed in {runtime:.2f} seconds")
        log("Artifacts saved to ./outputs/")

        # Print key results
        iso_recall_1 = iso_metrics["recall"]["1"]
        xgb_recall_1 = xgb_metrics["recall"]["1"]
        log(f"Fraud Recall - IsolationForest: {iso_recall_1:.4f}")
        log(f"Fraud Recall - SMOTE+XGBoost: {xgb_recall_1:.4f}")

    except Exception as e:
        log(f"Error: {str(e)}", logging.ERROR)
        raise


if __name__ == "__main__":
    main()
