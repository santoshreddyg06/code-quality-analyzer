"""
src/models/trainer.py
=====================
Model training, evaluation, and selection pipeline.

Trains and benchmarks:
1. Logistic Regression (standardized linear baseline)
2. Random Forest Classifier (bagging ensemble)
3. XGBoost Classifier (gradient boosting)

Selects the best model based on F1-Score / ROC-AUC and persists the trained artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

from src.config import cfg, settings
from src.features.feature_extractor import FEATURE_COLUMNS


def load_dataset(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test splits from CSV files."""
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Train/val/test splits not found in {processed_dir}. "
            "Please run 'src.pipeline.build_dataset' first."
        )

    return (
        pd.read_csv(train_path),
        pd.read_csv(val_path),
        pd.read_csv(test_path),
    )


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute standard classification evaluation metrics."""
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)[:, 1]
        auc = round(float(roc_auc_score(y, y_proba)), 4)
    except Exception:
        auc = 0.5

    return {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "roc_auc": auc,
    }


def train_and_evaluate_all() -> dict[str, Any]:
    """Train candidate models, compare performance on validation set, and persist the winner."""
    settings.ensure_data_dirs()
    processed_dir = settings.data_processed_dir
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_dataset(processed_dir)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["label"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    logger.info(f"Loaded dataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    best_name = None
    best_f1 = -1.0

    logger.info("\n" + "=" * 60)
    logger.info("MODEL TRAINING & VALIDATION COMPARISON")
    logger.info("=" * 60)

    for name, model in candidates.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        val_metrics = evaluate_model(model, X_val, y_val)
        logger.info(f"  {name} Val Metrics: {val_metrics}")
        results[name] = {"val_metrics": val_metrics, "model": model}

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_name = name

    logger.info(f"\nWinning Model: {best_name} (Validation F1 = {best_f1:.4f})")

    # Combine Train + Val to train final model, evaluate on Test set
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    final_model = candidates[best_name]
    final_model.fit(X_train_val, y_train_val)

    test_metrics = evaluate_model(final_model, X_test, y_test)
    logger.info(f"Final Test Metrics for {best_name}: {test_metrics}")

    # Persist model and artifacts
    model_path = models_dir / "best_model.joblib"
    joblib.dump(final_model, model_path)
    logger.info(f"Saved winning model to {model_path}")

    # Save feature names
    features_path = models_dir / "features.json"
    with features_path.open("w", encoding="utf-8") as fh:
        json.dump(FEATURE_COLUMNS, fh, indent=2)

    # Save metrics report
    report = {
        "best_model_name": best_name,
        "test_metrics": test_metrics,
        "candidate_val_metrics": {k: v["val_metrics"] for k, v in results.items()},
        "feature_count": len(FEATURE_COLUMNS),
        "train_samples": len(X_train_val),
        "test_samples": len(X_test),
    }
    metrics_path = models_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info(f"Saved evaluation report to {metrics_path}")

    return report


if __name__ == "__main__":
    train_and_evaluate_all()
