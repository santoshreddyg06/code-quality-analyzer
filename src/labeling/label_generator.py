"""
src/labeling/label_generator.py
===============================
Generates ground-truth maintainability labels for training data:
- "Likely Improvement" (Class 0)
- "Potential Degradation" (Class 1)

Ground truth combines:
1. Quantitative metric shifts (Radon MI drops, Cyclomatic spikes, Nesting depth)
2. Commit semantics (Reverts, bug-fixes, churn-to-comment ratios)
"""

from __future__ import annotations

from typing import Any

CLASS_IMPROVEMENT: int = 0
CLASS_DEGRADATION: int = 1

LABEL_NAMES: dict[int, str] = {
    CLASS_IMPROVEMENT: "Likely Improvement",
    CLASS_DEGRADATION: "Potential Degradation",
}


def compute_maintainability_label(feature_row: dict[str, Any]) -> dict[str, Any]:
    """
    Determine the ground-truth maintainability label for a commit feature row.

    Rules for 'Potential Degradation' (Class 1):
    1. Maintainability Index declines noticeably (delta_maintainability_index < -1.5)
    2. Complexity spikes significantly (delta_total_complexity >= 4 or delta_max_complexity >= 3)
    3. High-risk functions added (delta_high_complexity_functions >= 1)
    4. Deep nesting introduced (delta_max_nesting_depth >= 2)
    5. Revert commit (indicates previous code broke or degraded system)
    6. High churn with no comments and declining MI (churn > 100, net_comments <= 0, delta_mi < 0)

    Otherwise: 'Likely Improvement' (Class 0).

    Returns:
        Dict with 'label': int (0 or 1), 'label_name': str, and 'degradation_reasons': list[str].
    """
    reasons: list[str] = []

    delta_mi = float(feature_row.get("delta_maintainability_index", 0.0))
    delta_total_cc = int(feature_row.get("delta_total_complexity", 0))
    delta_max_cc = int(feature_row.get("delta_max_complexity", 0))
    delta_high_risk = int(feature_row.get("delta_high_complexity_functions", 0))
    delta_nesting = int(feature_row.get("delta_max_nesting_depth", 0))
    churn = int(feature_row.get("churn", 0))
    net_comments = int(feature_row.get("net_comments", 0))
    is_revert = bool(feature_row.get("msg_is_revert", False))

    if delta_mi < -1.5:
        reasons.append(f"Maintainability Index dropped by {abs(delta_mi):.1f} pts")

    if delta_total_cc >= 4:
        reasons.append(f"Total Cyclomatic Complexity surged by +{delta_total_cc}")

    if delta_max_cc >= 3:
        reasons.append(f"Max Cyclomatic Complexity jumped by +{delta_max_cc}")

    if delta_high_risk >= 1:
        reasons.append(f"Added {delta_high_risk} high-complexity function(s) (CC > 10)")

    if delta_nesting >= 2:
        reasons.append(f"Control flow nesting depth increased by +{delta_nesting}")

    if is_revert:
        reasons.append("Commit is a revert of previous problematic code")

    if churn > 100 and net_comments <= 0 and delta_mi < 0:
        reasons.append(f"High churn ({churn} lines) without documentation while MI dropped")

    is_degradation = len(reasons) > 0
    label = CLASS_DEGRADATION if is_degradation else CLASS_IMPROVEMENT

    return {
        "label": label,
        "label_name": LABEL_NAMES[label],
        "degradation_reasons": reasons,
    }
