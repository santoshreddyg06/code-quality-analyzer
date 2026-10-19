"""
scripts/run_pipeline.py
=======================
Master script to execute the end-to-end ML pipeline:
1. Verify raw data
2. Build processed dataset & splits
3. Train & evaluate candidate models
4. Select & persist winning model
5. Generate sample SHAP explanation
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.explainability.shap_explainer import ModelExplainer
from src.features.feature_extractor import FEATURE_COLUMNS
from src.models.predictor import QualityPredictor
from src.models.trainer import train_and_evaluate_all
from src.pipeline.build_dataset import build_dataset


def run_full_pipeline():
    logger.info("=" * 70)
    logger.info("STARTING END-TO-END ML PIPELINE")
    logger.info("=" * 70)

    # Step 1: Build Dataset
    logger.info("\n>>> Step 1: Processing raw commits into tabular dataset...")
    df = build_dataset()
    logger.info(f"Dataset successfully built with {len(df)} samples.")

    # Step 2: Train Models
    logger.info("\n>>> Step 2: Training & evaluating candidate ML models...")
    report = train_and_evaluate_all()
    logger.info(f"Model training complete. Winner: {report['best_model_name']}")

    # Step 3: Verify Inference Engine
    logger.info("\n>>> Step 3: Validating QualityPredictor inference engine...")
    predictor = QualityPredictor()
    sample_features = {col: 0.0 for col in FEATURE_COLUMNS}
    sample_features["delta_maintainability_index"] = -3.5
    sample_features["delta_total_complexity"] = 6
    sample_features["delta_max_nesting_depth"] = 2
    sample_features["churn"] = 45

    pred = predictor.predict(sample_features)
    logger.info(f"Sample Prediction: {pred}")

    # Step 4: Verify SHAP Explainer
    logger.info("\n>>> Step 4: Validating SHAP Explainer engine...")
    explainer = ModelExplainer()
    explanation = explainer.explain_prediction(sample_features)
    logger.info("SHAP Narrative:\n" + explanation["narrative"])

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
