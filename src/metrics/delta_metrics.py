"""
src/metrics/delta_metrics.py
============================
Computes change deltas between two versions of code (old vs new).

Delta = new_value - old_value

Interpretation:
- delta_maintainability_index > 0: Maintainability improved (+)
- delta_maintainability_index < 0: Maintainability degraded (-)
- delta_total_complexity > 0: Complexity increased (often increases risk)
- delta_max_nesting_depth > 0: Deep nested logic added
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.metrics.code_metrics import CodeMetrics


@dataclass
class DeltaMetrics:
    """Quantitative differences between new code metrics and old code metrics."""
    delta_loc: int = 0
    delta_sloc: int = 0
    delta_lloc: int = 0
    delta_comments: int = 0
    delta_comment_ratio: float = 0.0

    delta_total_complexity: int = 0
    delta_avg_complexity: float = 0.0
    delta_max_complexity: int = 0
    delta_high_complexity_functions: int = 0

    delta_maintainability_index: float = 0.0
    delta_halstead_volume: float = 0.0
    delta_halstead_difficulty: float = 0.0
    delta_halstead_effort: float = 0.0
    delta_halstead_bugs: float = 0.0

    delta_lizard_ccn: int = 0
    delta_lizard_token_count: int = 0
    delta_lizard_max_ccn: int = 0

    delta_functions_count: int = 0
    delta_classes_count: int = 0
    delta_max_nesting_depth: int = 0
    delta_returns_count: int = 0
    delta_raises_count: int = 0
    delta_try_except_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to plain dictionary."""
        return asdict(self)


def compute_delta_metrics(old_m: CodeMetrics, new_m: CodeMetrics) -> DeltaMetrics:
    """
    Compute element-wise deltas: new_metrics - old_metrics.

    Args:
        old_m: CodeMetrics of the previous/parent version.
        new_m: CodeMetrics of the modified/new version.

    Returns:
        DeltaMetrics dataclass with all deltas calculated.
    """
    return DeltaMetrics(
        delta_loc=new_m.loc - old_m.loc,
        delta_sloc=new_m.sloc - old_m.sloc,
        delta_lloc=new_m.lloc - old_m.lloc,
        delta_comments=new_m.comments - old_m.comments,
        delta_comment_ratio=round(new_m.comment_ratio - old_m.comment_ratio, 4),

        delta_total_complexity=new_m.total_complexity - old_m.total_complexity,
        delta_avg_complexity=round(new_m.avg_complexity - old_m.avg_complexity, 2),
        delta_max_complexity=new_m.max_complexity - old_m.max_complexity,
        delta_high_complexity_functions=new_m.high_complexity_functions - old_m.high_complexity_functions,

        delta_maintainability_index=round(new_m.maintainability_index - old_m.maintainability_index, 2),
        delta_halstead_volume=round(new_m.halstead_volume - old_m.halstead_volume, 2),
        delta_halstead_difficulty=round(new_m.halstead_difficulty - old_m.halstead_difficulty, 2),
        delta_halstead_effort=round(new_m.halstead_effort - old_m.halstead_effort, 2),
        delta_halstead_bugs=round(new_m.halstead_bugs - old_m.halstead_bugs, 3),

        delta_lizard_ccn=new_m.lizard_ccn - old_m.lizard_ccn,
        delta_lizard_token_count=new_m.lizard_token_count - old_m.lizard_token_count,
        delta_lizard_max_ccn=new_m.lizard_max_ccn - old_m.lizard_max_ccn,

        delta_functions_count=new_m.functions_count - old_m.functions_count,
        delta_classes_count=new_m.classes_count - old_m.classes_count,
        delta_max_nesting_depth=new_m.max_nesting_depth - old_m.max_nesting_depth,
        delta_returns_count=new_m.returns_count - old_m.returns_count,
        delta_raises_count=new_m.raises_count - old_m.raises_count,
        delta_try_except_count=new_m.try_except_count - old_m.try_except_count,
    )
