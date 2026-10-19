"""
src/labeling
============
Ground-truth maintainability labeling package.
"""

from src.labeling.label_generator import (
    CLASS_DEGRADATION,
    CLASS_IMPROVEMENT,
    LABEL_NAMES,
    compute_maintainability_label,
)

__all__ = [
    "CLASS_IMPROVEMENT",
    "CLASS_DEGRADATION",
    "LABEL_NAMES",
    "compute_maintainability_label",
]
