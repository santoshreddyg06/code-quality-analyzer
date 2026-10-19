"""
src/models/predictor.py
=======================
Inference engine that loads the trained model artifact and serves predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd
import joblib

from src.features.feature_extractor import FEATURE_COLUMNS
from src.labeling.label_generator import LABEL_NAMES, CLASS_DEGRADATION, CLASS_IMPROVEMENT


class QualityPredictor:
    """Production quality prediction wrapper."""

    def __init__(self, model_path: str | Path = "models/best_model.joblib") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. Train a model first."
            )
        self.model = joblib.load(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")

    def predict(self, features: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        """
        Predict maintainability impact from features.

        Args:
            features: Dictionary or DataFrame with columns matching FEATURE_COLUMNS.

        Returns:
            Dict containing prediction label, probabilities, and risk level.
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])[FEATURE_COLUMNS]
        else:
            df = features[FEATURE_COLUMNS]

        pred_code = int(self.model.predict(df)[0])
        pred_label = LABEL_NAMES.get(pred_code, "Unknown")

        try:
            probabilities = self.model.predict_proba(df)[0]
            prob_improvement = round(float(probabilities[CLASS_IMPROVEMENT]), 4)
            prob_degradation = round(float(probabilities[CLASS_DEGRADATION]), 4)
        except Exception:
            prob_degradation = 1.0 if pred_code == CLASS_DEGRADATION else 0.0
            prob_improvement = 1.0 - prob_degradation

        # Risk classification
        if prob_degradation >= 0.70:
            risk_level = "High Risk"
        elif prob_degradation >= 0.40:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        return {
            "prediction_code": pred_code,
            "prediction": pred_label,
            "degradation_probability": prob_degradation,
            "improvement_probability": prob_improvement,
            "risk_level": risk_level,
        }
