"""
tests/test_metrics.py
=====================
Unit tests for the metrics extraction modules:
- code_metrics
- diff_metrics
- delta_metrics
"""

from __future__ import annotations

import pytest

from src.metrics.code_metrics import CodeMetrics, extract_code_metrics
from src.metrics.diff_metrics import (
    DiffMetrics,
    extract_diff_metrics,
    reconstruct_code_versions_from_patch,
)
from src.metrics.delta_metrics import DeltaMetrics, compute_delta_metrics


SAMPLE_PYTHON_CODE = """
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
"""

SAMPLE_PATCH = """@@ -1,5 +1,8 @@
 def compute(x):
-    return x * 2
+    if x < 0:
+        raise ValueError("Must be non-negative")
+    # Double the input
+    return x * 2
"""


class TestCodeMetrics:
    def test_extract_code_metrics_valid(self):
        m = extract_code_metrics(SAMPLE_PYTHON_CODE)
        assert isinstance(m, CodeMetrics)
        assert m.loc > 0
        assert m.sloc > 0
        assert m.classes_count == 1
        assert m.functions_count == 2
        assert m.returns_count == 2
        assert m.raises_count == 1
        assert m.total_complexity >= 2
        assert m.maintainability_index > 0

    def test_extract_code_metrics_empty(self):
        m = extract_code_metrics("")
        assert m.loc == 0
        assert m.total_complexity == 0
        assert m.maintainability_index == 100.0

    def test_extract_code_metrics_syntax_error(self):
        # Incomplete snippet should not raise an unhandled exception
        bad_code = "def broken_func(:\n    pass\n"
        m = extract_code_metrics(bad_code)
        assert isinstance(m, CodeMetrics)


class TestDiffMetrics:
    def test_extract_diff_metrics_valid(self):
        d = extract_diff_metrics(SAMPLE_PATCH)
        assert isinstance(d, DiffMetrics)
        assert d.lines_added == 4
        assert d.lines_deleted == 1
        assert d.net_lines == 3
        assert d.churn == 5
        assert d.hunks_count == 1
        assert d.added_branch_keywords >= 1  # "if"
        assert d.added_error_handlers >= 1   # "raise"
        assert d.added_comments >= 1

    def test_extract_diff_metrics_empty(self):
        d = extract_diff_metrics("")
        assert d.lines_added == 0
        assert d.lines_deleted == 0
        assert d.churn == 0

    def test_reconstruct_code_versions_from_patch(self):
        old_code, new_code = reconstruct_code_versions_from_patch(SAMPLE_PATCH)
        assert "return x * 2" in old_code
        assert "if x < 0:" not in old_code
        assert "if x < 0:" in new_code
        assert "raise ValueError" in new_code


class TestDeltaMetrics:
    def test_compute_delta_metrics(self):
        old_m = CodeMetrics(
            loc=50,
            sloc=40,
            total_complexity=5,
            maintainability_index=85.0,
            functions_count=3,
        )
        new_m = CodeMetrics(
            loc=70,
            sloc=55,
            total_complexity=8,
            maintainability_index=78.0,
            functions_count=4,
        )
        delta = compute_delta_metrics(old_m, new_m)
        assert delta.delta_loc == 20
        assert delta.delta_sloc == 15
        assert delta.delta_total_complexity == 3
        assert delta.delta_maintainability_index == -7.0
        assert delta.delta_functions_count == 1
