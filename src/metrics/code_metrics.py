"""
src/metrics/code_metrics.py
===========================
Extracts static code quality metrics from Python code strings using:
1. Radon (Cyclomatic Complexity, Maintainability Index, Halstead Metrics, Raw LOC)
2. Lizard (Token counts, CCN)
3. Python AST (Nesting depth, function/class counts, structure)

All extractors are resilient: if a snippet has syntax errors or is partial,
safe defaults are returned without failing.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any

from loguru import logger
import lizard
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
import radon.raw as radon_raw


@dataclass
class CodeMetrics:
    """Structured code quality metrics for a single Python source snippet."""
    # Line counts
    loc: int = 0
    lloc: int = 0
    sloc: int = 0
    comments: int = 0
    blank_lines: int = 0
    comment_ratio: float = 0.0

    # Radon Cyclomatic Complexity
    total_complexity: int = 0
    avg_complexity: float = 0.0
    max_complexity: int = 0
    high_complexity_functions: int = 0  # CC > 10

    # Radon Maintainability & Halstead
    maintainability_index: float = 100.0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    halstead_bugs: float = 0.0

    # Lizard Metrics
    lizard_ccn: int = 0
    lizard_nloc: int = 0
    lizard_token_count: int = 0
    lizard_max_ccn: int = 0

    # AST Metrics
    functions_count: int = 0
    classes_count: int = 0
    max_nesting_depth: int = 0
    returns_count: int = 0
    raises_count: int = 0
    try_except_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to plain dictionary."""
        return asdict(self)


class _ASTVisitor(ast.NodeVisitor):
    """Walks the Python AST to extract structural complexity metrics."""

    def __init__(self) -> None:
        self.functions_count = 0
        self.classes_count = 0
        self.returns_count = 0
        self.raises_count = 0
        self.try_except_count = 0
        self.max_nesting = 0
        self._current_nesting = 0

    def _track_nesting(self, node: ast.AST) -> None:
        nesting_nodes = (
            ast.If,
            ast.For,
            ast.While,
            ast.Try,
            ast.With,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        )
        is_nesting = isinstance(node, nesting_nodes)
        if is_nesting:
            self._current_nesting += 1
            if self._current_nesting > self.max_nesting:
                self.max_nesting = self._current_nesting

        self.generic_visit(node)

        if is_nesting:
            self._current_nesting -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions_count += 1
        self._track_nesting(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions_count += 1
        self._track_nesting(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes_count += 1
        self._track_nesting(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.returns_count += 1
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.raises_count += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.try_except_count += 1
        self._track_nesting(node)

    def visit_If(self, node: ast.If) -> None:
        self._track_nesting(node)

    def visit_For(self, node: ast.For) -> None:
        self._track_nesting(node)

    def visit_While(self, node: ast.While) -> None:
        self._track_nesting(node)

    def visit_With(self, node: ast.With) -> None:
        self._track_nesting(node)


def extract_code_metrics(code: str) -> CodeMetrics:
    """
    Extract comprehensive code metrics from Python source text.

    Args:
        code: Python source code string.

    Returns:
        CodeMetrics dataclass populated with all extracted values.
    """
    if not code or not code.strip():
        return CodeMetrics()

    metrics = CodeMetrics()

    # 1. Radon Raw metrics (LOC, SLOC, comments, blank lines)
    try:
        raw_res = radon_raw.analyze(code)
        metrics.loc = raw_res.loc
        metrics.lloc = raw_res.lloc
        metrics.sloc = raw_res.sloc
        metrics.comments = raw_res.comments
        metrics.blank_lines = raw_res.blank
        total_non_blank = raw_res.loc - raw_res.blank
        if total_non_blank > 0:
            metrics.comment_ratio = round(raw_res.comments / total_non_blank, 4)
    except Exception as exc:
        logger.debug(f"Radon raw analysis skipped: {exc}")
        lines = code.splitlines()
        metrics.loc = len(lines)
        metrics.sloc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

    # 2. Radon Cyclomatic Complexity
    try:
        cc_blocks = radon_cc.cc_visit(code)
        if cc_blocks:
            complexities = [b.complexity for b in cc_blocks]
            metrics.total_complexity = sum(complexities)
            metrics.avg_complexity = round(sum(complexities) / len(complexities), 2)
            metrics.max_complexity = max(complexities)
            metrics.high_complexity_functions = sum(1 for c in complexities if c > 10)
    except Exception as exc:
        logger.debug(f"Radon CC analysis skipped: {exc}")

    # 3. Radon Maintainability Index & Halstead
    try:
        metrics.maintainability_index = round(radon_metrics.mi_visit(code, multi=True), 2)
    except Exception as exc:
        logger.debug(f"Radon MI analysis skipped: {exc}")

    try:
        h_res = radon_metrics.h_visit(code)
        metrics.halstead_volume = round(getattr(h_res.total, "volume", 0.0), 2)
        metrics.halstead_difficulty = round(getattr(h_res.total, "difficulty", 0.0), 2)
        metrics.halstead_effort = round(getattr(h_res.total, "effort", 0.0), 2)
        metrics.halstead_bugs = round(getattr(h_res.total, "bugs", 0.0), 3)
    except Exception as exc:
        logger.debug(f"Radon Halstead analysis skipped: {exc}")

    # 4. Lizard analysis
    try:
        liz_res = lizard.analyze_file.analyze_source_code("snippet.py", code)
        metrics.lizard_ccn = liz_res.CCN
        metrics.lizard_nloc = liz_res.nloc
        metrics.lizard_token_count = liz_res.token_count
        if liz_res.function_list:
            metrics.lizard_max_ccn = max(f.cyclomatic_complexity for f in liz_res.function_list)
    except Exception as exc:
        logger.debug(f"Lizard analysis skipped: {exc}")

    # 5. AST analysis
    try:
        tree = ast.parse(code)
        visitor = _ASTVisitor()
        visitor.visit(tree)
        metrics.functions_count = visitor.functions_count
        metrics.classes_count = visitor.classes_count
        metrics.max_nesting_depth = visitor.max_nesting
        metrics.returns_count = visitor.returns_count
        metrics.raises_count = visitor.raises_count
        metrics.try_except_count = visitor.try_except_count
    except SyntaxError:
        logger.debug("Code snippet could not be parsed into AST (partial syntax).")
    except Exception as exc:
        logger.debug(f"AST analysis skipped: {exc}")

    return metrics
