"""
tests/test_features_labeling.py
===============================
Unit tests for feature extraction and labeling.
"""

from __future__ import annotations

import pytest

from src.features.feature_extractor import (
    FEATURE_COLUMNS,
    extract_features_from_commit_record,
    extract_features_from_files,
)
from src.labeling.label_generator import (
    CLASS_DEGRADATION,
    CLASS_IMPROVEMENT,
    compute_maintainability_label,
)


def test_extract_features_from_files_empty():
    features = extract_features_from_files([])
    assert isinstance(features, dict)
    for col in FEATURE_COLUMNS:
        assert col in features


def test_extract_features_from_patch():
    file_change = {
        "filename": "service.py",
        "additions": 10,
        "deletions": 2,
        "patch": "@@ -1,5 +1,10 @@\n def run():\n+    if True:\n+        print('hello')\n     pass",
    }
    features = extract_features_from_files([file_change])
    assert features["files_changed_count"] == 1
    assert features["lines_added"] >= 2
    assert features["churn"] >= 2
    assert "delta_maintainability_index" in features


def test_labeling_improvement():
    row = {
        "delta_maintainability_index": 5.0,
        "delta_total_complexity": 0,
        "delta_max_complexity": 0,
        "delta_high_complexity_functions": 0,
        "delta_max_nesting_depth": 0,
        "churn": 20,
        "net_comments": 2,
        "msg_is_revert": False,
    }
    label_info = compute_maintainability_label(row)
    assert label_info["label"] == CLASS_IMPROVEMENT
    assert label_info["label_name"] == "Likely Improvement"
    assert len(label_info["degradation_reasons"]) == 0


def test_labeling_degradation_due_to_complexity():
    row = {
        "delta_maintainability_index": -4.0,
        "delta_total_complexity": 8,
        "delta_max_complexity": 5,
        "delta_high_complexity_functions": 2,
        "delta_max_nesting_depth": 3,
        "churn": 150,
        "net_comments": -5,
        "msg_is_revert": False,
    }
    label_info = compute_maintainability_label(row)
    assert label_info["label"] == CLASS_DEGRADATION
    assert label_info["label_name"] == "Potential Degradation"
    assert len(label_info["degradation_reasons"]) > 0
