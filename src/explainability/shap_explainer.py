"""
src/explainability/shap_explainer.py
===================================
SHAP (SHapley Additive exPlanations) explainability engine.

Generates:
1. Feature-level SHAP values for individual commit predictions
2. Human-readable English explanations for software developers
3. Global feature importance rankings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
import shap
import joblib

from src.features.feature_extractor import FEATURE_COLUMNS
from src.labeling.label_generator import LABEL_NAMES


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "delta_maintainability_index": "Maintainability Index shift",
    "delta_total_complexity": "Total Cyclomatic Complexity change",
    "delta_avg_complexity": "Average function complexity change",
    "delta_max_complexity": "Peak function complexity change",
    "delta_high_complexity_functions": "Change in high-risk functions (CC > 10)",
    "delta_halstead_volume": "Halstead code volume delta",
    "delta_halstead_effort": "Cognitive mental effort delta",
    "delta_max_nesting_depth": "Maximum control flow nesting depth change",
    "delta_classes_count": "Classes added or removed",
    "delta_functions_count": "Functions added or removed",
    "delta_try_except_count": "Exception handlers added/removed",
    "net_branch_keywords": "Net branch statements added (if/for/while)",
    "net_error_handlers": "Net error handling blocks added",
    "net_comments": "Net comment lines added",
    "churn": "Total code churn (added + deleted lines)",
    "lines_added": "Lines of code added",
    "lines_deleted": "Lines of code deleted",
    "net_lines": "Net change in lines of code",
    "deletion_ratio": "Proportion of churn that is deletions",
    "files_changed_count": "Number of Python files touched",
    "hunks_count": "Number of separate diff hunks",
    "old_maintainability_index": "Original Maintainability Index",
    "old_total_complexity": "Original Cyclomatic Complexity",
    "new_maintainability_index": "Resulting Maintainability Index",
    "new_total_complexity": "Resulting Cyclomatic Complexity",
}


class ModelExplainer:
    """SHAP Explainer wrapper for code quality models."""

    def __init__(
        self,
        model_path: str | Path = "models/best_model.joblib",
        background_data_path: str | Path = "data/processed/train.csv",
    ) -> None:
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)

        # Extract underlying classifier if inside a pipeline
        if hasattr(self.model, "named_steps"):
            self.clf = self.model.named_steps.get("clf", self.model)
        else:
            self.clf = self.model

        self.background_df: pd.DataFrame | None = None
        bg_path = Path(background_data_path)
        if bg_path.exists():
            try:
                bg = pd.read_csv(bg_path)[FEATURE_COLUMNS]
                self.background_df = bg.sample(min(100, len(bg)), random_state=42)
            except Exception as e:
                logger.warning(f"Could not load background data: {e}")

        # Initialize SHAP explainer
        try:
            self.explainer = shap.TreeExplainer(self.clf)
        except Exception:
            # Fall back to general Explainer (e.g. for linear models)
            if self.background_df is not None:
                self.explainer = shap.Explainer(self.clf, self.background_df)
            else:
                self.explainer = shap.Explainer(self.clf)

        logger.info("SHAP Explainer initialized successfully.")

    def explain_prediction(self, features: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        """
        Explain a single prediction with SHAP values and human-readable insights.

        Args:
            features: Feature dictionary or single-row DataFrame.

        Returns:
            Dict containing base value, sorted feature contributions, and plain-English summary.
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])[FEATURE_COLUMNS]
        else:
            df = features[FEATURE_COLUMNS]

        shap_values_raw = self.explainer(df)

        # Handle binary classification shapes (e.g., shape (1, features) or (1, features, 2))
        values = shap_values_raw.values
        if len(values.shape) == 3:
            # Multi-class output: select degradation class (index 1)
            row_shap = values[0, :, 1]
            base_val = float(shap_values_raw.base_values[0, 1])
        else:
            row_shap = values[0]
            base_val = float(shap_values_raw.base_values[0])

        contributions: list[dict[str, Any]] = []
        for feat, val, s_val in zip(FEATURE_COLUMNS, df.iloc[0], row_shap):
            s_float = round(float(s_val), 4)
            contributions.append({
                "feature": feat,
                "display_name": FEATURE_DESCRIPTIONS.get(feat, feat),
                "feature_value": round(float(val), 2),
                "shap_value": s_float,
                "direction": "Increases Risk" if s_float > 0 else "Decreases Risk",
            })

        # Sort by absolute SHAP magnitude
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        top_risk_factors = [c for c in contributions if c["shap_value"] > 0][:4]
        top_positive_factors = [c for c in contributions if c["shap_value"] < 0][:4]

        # Generate readable narrative summary
        summary_lines = []
        if top_risk_factors:
            summary_lines.append("**Factors increasing maintainability risk:**")
            for f in top_risk_factors:
                summary_lines.append(
                    f"- {f['display_name']} ({f['feature_value']}): contributed +{f['shap_value']:.3f} to degradation probability"
                )

        if top_positive_factors:
            summary_lines.append("\n**Factors supporting healthy maintainability:**")
            for f in top_positive_factors:
                summary_lines.append(
                    f"- {f['display_name']} ({f['feature_value']}): reduced degradation risk by {abs(f['shap_value']):.3f}"
                )

        narrative = "\n".join(summary_lines)

        return {
            "base_value": round(base_val, 4),
            "contributions": contributions,
            "top_risk_factors": top_risk_factors,
            "top_positive_factors": top_positive_factors,
            "narrative": narrative,
        }
