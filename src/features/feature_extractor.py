"""
src/features/feature_extractor.py
=================================
Transforms raw commit records or file change pairs into a consistent,
standardized feature vector for machine learning and SHAP explanations.
"""

from __future__ import annotations

from typing import Any

from src.metrics.code_metrics import CodeMetrics, extract_code_metrics
from src.metrics.delta_metrics import DeltaMetrics, compute_delta_metrics
from src.metrics.diff_metrics import (
    DiffMetrics,
    extract_diff_metrics,
    reconstruct_code_versions_from_patch,
)

# Ordered feature column names used across training, evaluation, and inference
FEATURE_COLUMNS: list[str] = [
    "files_changed_count",
    "lines_added",
    "lines_deleted",
    "net_lines",
    "churn",
    "deletion_ratio",
    "hunks_count",
    "delta_maintainability_index",
    "delta_total_complexity",
    "delta_avg_complexity",
    "delta_max_complexity",
    "delta_high_complexity_functions",
    "delta_halstead_volume",
    "delta_halstead_effort",
    "delta_max_nesting_depth",
    "delta_classes_count",
    "delta_functions_count",
    "delta_try_except_count",
    "net_branch_keywords",
    "net_error_handlers",
    "net_comments",
    "old_maintainability_index",
    "old_total_complexity",
    "new_maintainability_index",
    "new_total_complexity",
]


def extract_features_from_files(
    file_changes: list[dict[str, Any]],
) -> dict[str, float | int]:
    """
    Extract aggregated commit-level features from a list of changed files.

    Each file dict should have either:
      - 'patch': unified diff string
    OR:
      - 'old_code': str and 'new_code': str

    Args:
        file_changes: List of file change dicts.

    Returns:
        Dict mapping each feature in FEATURE_COLUMNS to its numeric value.
    """
    if not file_changes:
        return {col: 0.0 for col in FEATURE_COLUMNS}

    files_count = len(file_changes)
    total_added = 0
    total_deleted = 0
    total_hunks = 0

    net_branch_kw = 0
    net_error_kw = 0
    net_comments = 0

    old_metrics_list: list[CodeMetrics] = []
    new_metrics_list: list[CodeMetrics] = []
    deltas_list: list[DeltaMetrics] = []

    for file_info in file_changes:
        patch: str | None = file_info.get("patch")
        old_code: str = file_info.get("old_code", "")
        new_code: str = file_info.get("new_code", "")

        # If full code is not provided but patch is, reconstruct approximations
        if (not old_code and not new_code) and patch:
            old_code, new_code = reconstruct_code_versions_from_patch(patch)

        # Diff metrics
        diff_m = extract_diff_metrics(patch) if patch else DiffMetrics(
            lines_added=file_info.get("additions", 0),
            lines_deleted=file_info.get("deletions", 0),
            net_lines=file_info.get("additions", 0) - file_info.get("deletions", 0),
            churn=file_info.get("additions", 0) + file_info.get("deletions", 0),
        )

        total_added += diff_m.lines_added
        total_deleted += diff_m.lines_deleted
        total_hunks += diff_m.hunks_count
        net_branch_kw += diff_m.net_branch_keywords
        net_error_kw += diff_m.net_error_handlers
        net_comments += diff_m.net_comments

        # Code & Delta metrics
        old_m = extract_code_metrics(old_code)
        new_m = extract_code_metrics(new_code)
        delta = compute_delta_metrics(old_m, new_m)

        old_metrics_list.append(old_m)
        new_metrics_list.append(new_m)
        deltas_list.append(delta)

    churn = total_added + total_deleted
    deletion_ratio = round(total_deleted / churn, 4) if churn > 0 else 0.0

    # Aggregations across files
    delta_mi = round(sum(d.delta_maintainability_index for d in deltas_list) / files_count, 2)
    delta_total_cc = sum(d.delta_total_complexity for d in deltas_list)
    delta_avg_cc = round(sum(d.delta_avg_complexity for d in deltas_list) / files_count, 2)
    delta_max_cc = max((d.delta_max_complexity for d in deltas_list), default=0)
    delta_high_risk = sum(d.delta_high_complexity_functions for d in deltas_list)
    delta_h_vol = round(sum(d.delta_halstead_volume for d in deltas_list), 2)
    delta_h_effort = round(sum(d.delta_halstead_effort for d in deltas_list), 2)
    delta_max_nesting = max((d.delta_max_nesting_depth for d in deltas_list), default=0)
    delta_classes = sum(d.delta_classes_count for d in deltas_list)
    delta_funcs = sum(d.delta_functions_count for d in deltas_list)
    delta_try = sum(d.delta_try_except_count for d in deltas_list)

    old_mi = round(sum(m.maintainability_index for m in old_metrics_list) / files_count, 2)
    old_cc = sum(m.total_complexity for m in old_metrics_list)
    new_mi = round(sum(m.maintainability_index for m in new_metrics_list) / files_count, 2)
    new_cc = sum(m.total_complexity for m in new_metrics_list)

    return {
        "files_changed_count": files_count,
        "lines_added": total_added,
        "lines_deleted": total_deleted,
        "net_lines": total_added - total_deleted,
        "churn": churn,
        "deletion_ratio": deletion_ratio,
        "hunks_count": total_hunks,
        "delta_maintainability_index": delta_mi,
        "delta_total_complexity": delta_total_cc,
        "delta_avg_complexity": delta_avg_cc,
        "delta_max_complexity": delta_max_cc,
        "delta_high_complexity_functions": delta_high_risk,
        "delta_halstead_volume": delta_h_vol,
        "delta_halstead_effort": delta_h_effort,
        "delta_max_nesting_depth": delta_max_nesting,
        "delta_classes_count": delta_classes,
        "delta_functions_count": delta_funcs,
        "delta_try_except_count": delta_try,
        "net_branch_keywords": net_branch_kw,
        "net_error_handlers": net_error_kw,
        "net_comments": net_comments,
        "old_maintainability_index": old_mi,
        "old_total_complexity": old_cc,
        "new_maintainability_index": new_mi,
        "new_total_complexity": new_cc,
    }


def extract_features_from_commit_record(commit_record: dict[str, Any]) -> dict[str, Any]:
    """
    Extract ML features from a stored historical commit record.

    Returns a dict containing commit metadata (sha, repo, date) plus all features.
    """
    py_files = commit_record.get("files", [])
    features = extract_features_from_files(py_files)

    return {
        "sha": commit_record.get("sha", ""),
        "repo": commit_record.get("repo", ""),
        "authored_at": commit_record.get("authored_at", ""),
        "message": commit_record.get("message", ""),
        "msg_is_bugfix": commit_record.get("msg_is_bugfix", False),
        "msg_is_revert": commit_record.get("msg_is_revert", False),
        "msg_is_test": commit_record.get("msg_is_test", False),
        "msg_is_docs": commit_record.get("msg_is_docs", False),
        **features,
    }
