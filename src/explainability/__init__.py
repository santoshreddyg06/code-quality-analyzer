"""
src/explainability
==================
SHAP explainability package.
"""

from src.explainability.shap_explainer import (
    FEATURE_DESCRIPTIONS,
    ModelExplainer,
)

__all__ = [
    "FEATURE_DESCRIPTIONS",
    "ModelExplainer",
]
