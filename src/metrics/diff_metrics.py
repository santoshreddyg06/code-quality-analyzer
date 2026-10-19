"""
src/metrics/diff_metrics.py
===========================
Extracts structural and semantic metrics directly from patch/diff text.

When comparing code changes, diff metrics capture the "dynamics" of the edit:
- How much code was churned?
- Were branches added or removed?
- Was error handling added?
- Are comments being added or stripped?
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


_BRANCH_KEYWORDS_RE = re.compile(r"\b(if|elif|else|for|while|match|case)\b")
_ERROR_KEYWORDS_RE = re.compile(r"\b(try|except|finally|raise|assert)\b")
_FUNC_CLASS_RE = re.compile(r"\b(def|class|async\s+def)\b")
_COMMENT_RE = re.compile(r"^\s*#")


@dataclass
class DiffMetrics:
    """Metrics extracted directly from a unified diff / patch."""
    lines_added: int = 0
    lines_deleted: int = 0
    net_lines: int = 0
    churn: int = 0
    deletion_ratio: float = 0.0
    hunks_count: int = 0

    # Keywords in added vs deleted lines
    added_branch_keywords: int = 0
    deleted_branch_keywords: int = 0
    net_branch_keywords: int = 0

    added_error_handlers: int = 0
    deleted_error_handlers: int = 0
    net_error_handlers: int = 0

    added_definitions: int = 0
    deleted_definitions: int = 0
    net_definitions: int = 0

    added_comments: int = 0
    deleted_comments: int = 0
    net_comments: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to plain dictionary."""
        return asdict(self)


def extract_diff_metrics(patch: str | None) -> DiffMetrics:
    """
    Parse a unified diff string and return quantitative metrics.

    Args:
        patch: Unified diff string (such as GitHub commit patch).

    Returns:
        DiffMetrics dataclass.
    """
    if not patch or not patch.strip():
        return DiffMetrics()

    metrics = DiffMetrics()
    lines = patch.splitlines()

    added_lines_text: list[str] = []
    deleted_lines_text: list[str] = []

    for line in lines:
        if line.startswith("@@"):
            metrics.hunks_count += 1
        elif line.startswith("+") and not line.startswith("+++"):
            code = line[1:].strip()
            metrics.lines_added += 1
            if code:
                added_lines_text.append(code)
                if _COMMENT_RE.match(code):
                    metrics.added_comments += 1
                metrics.added_branch_keywords += len(_BRANCH_KEYWORDS_RE.findall(code))
                metrics.added_error_handlers += len(_ERROR_KEYWORDS_RE.findall(code))
                metrics.added_definitions += len(_FUNC_CLASS_RE.findall(code))

        elif line.startswith("-") and not line.startswith("---"):
            code = line[1:].strip()
            metrics.lines_deleted += 1
            if code:
                deleted_lines_text.append(code)
                if _COMMENT_RE.match(code):
                    metrics.deleted_comments += 1
                metrics.deleted_branch_keywords += len(_BRANCH_KEYWORDS_RE.findall(code))
                metrics.deleted_error_handlers += len(_ERROR_KEYWORDS_RE.findall(code))
                metrics.deleted_definitions += len(_FUNC_CLASS_RE.findall(code))

    metrics.net_lines = metrics.lines_added - metrics.lines_deleted
    metrics.churn = metrics.lines_added + metrics.lines_deleted
    if metrics.churn > 0:
        metrics.deletion_ratio = round(metrics.lines_deleted / metrics.churn, 4)

    metrics.net_branch_keywords = metrics.added_branch_keywords - metrics.deleted_branch_keywords
    metrics.net_error_handlers = metrics.added_error_handlers - metrics.deleted_error_handlers
    metrics.net_definitions = metrics.added_definitions - metrics.deleted_definitions
    metrics.net_comments = metrics.added_comments - metrics.deleted_comments

    return metrics


def reconstruct_code_versions_from_patch(patch: str | None) -> tuple[str, str]:
    """
    Reconstruct approximate old and new code representations from a unified diff patch.

    Lines with '+' are part of the new code.
    Lines with '-' are part of the old code.
    Context lines (starting with space or no +/-) are part of both.

    Returns:
        (old_code, new_code) tuple of strings.
    """
    if not patch:
        return "", ""

    old_lines: list[str] = []
    new_lines: list[str] = []

    for line in patch.splitlines():
        if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
            continue

        if line.startswith("+"):
            new_lines.append(line[1:])
        elif line.startswith("-"):
            old_lines.append(line[1:])
        else:
            # Context line (may start with ' ' or nothing)
            content = line[1:] if line.startswith(" ") else line
            old_lines.append(content)
            new_lines.append(content)

    return "\n".join(old_lines), "\n".join(new_lines)
