# -*- coding: utf-8 -*-
"""
src/data_collection/commit_collector.py
========================================
Collects historical commits from a GitHub repository and extracts
per-commit metadata and file-level diff information.

What we collect per commit
---------------------------
- SHA, message, author, date
- Number of files changed
- Lines added and deleted (overall)
- Per-file: filename, status (added/modified/removed), patch (the actual diff),
  lines added, lines deleted
- Whether the commit message suggests it is a bug fix or revert
  (used later in Phase 4 label generation)

What we intentionally do NOT collect
--------------------------------------
- Future information (bugs filed after the commit, CI results after merge)
  That would cause data leakage. Future signals are only used to BUILD labels,
  never as model input features.

Usage:
    from src.data_collection.commit_collector import CommitCollector
    collector = CommitCollector(repo_full_name="pallets/flask")
    records = collector.collect(max_commits=200)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from github import Github, Repository
from github.GithubException import GithubException

from src.data_collection.github_client import get_github_client, safe_github_call
from src.config import cfg

# --------------------------------------------------------------------------
# Regex patterns used to detect bug-fix and revert commits by message
# (heuristic only - used for labeling, not as features)
# --------------------------------------------------------------------------
_BUG_FIX_PATTERN = re.compile(
    r"\b(fix|bug|error|issue|patch|defect|regression|crash|fail|broken|hotfix)\b",
    re.IGNORECASE,
)
_REVERT_PATTERN = re.compile(r"\brevert\b", re.IGNORECASE)
_TEST_PATTERN   = re.compile(r"\b(test|spec|unittest)\b", re.IGNORECASE)
_DOCS_PATTERN   = re.compile(r"\b(doc|docs|readme|changelog|comment)\b", re.IGNORECASE)


def _classify_commit_message(message: str) -> dict[str, bool]:
    """
    Apply simple keyword heuristics to the commit message.

    Returns a dict of boolean flags. These are stored as metadata for
    later use in label generation - NOT used as ML features directly.
    """
    first_line = message.splitlines()[0] if message else ""
    return {
        "msg_is_bugfix": bool(_BUG_FIX_PATTERN.search(first_line)),
        "msg_is_revert": bool(_REVERT_PATTERN.search(first_line)),
        "msg_is_test":   bool(_TEST_PATTERN.search(first_line)),
        "msg_is_docs":   bool(_DOCS_PATTERN.search(first_line)),
    }


def _extract_file_record(f) -> dict[str, Any] | None:
    """
    Extract per-file information from a GitHub commit file object.

    Returns None for files we cannot meaningfully analyse (binary files,
    files with no patch, files that are not Python source).

    Why only Python files?
    - Our metric extraction tools (radon, AST) work on Python.
    - Mixing in YAML/JSON/Markdown changes adds noise with no signal.
    """
    filename: str = f.filename

    # Skip non-Python files
    if not filename.endswith(".py"):
        return None

    patch: str | None = getattr(f, "patch", None)

    return {
        "filename": filename,
        "status":   f.status,
        "additions": f.additions,
        "deletions": f.deletions,
        "changes":  f.changes,
        "patch":    patch,
    }


class CommitCollector:
    """
    Fetches commits from one GitHub repository and returns structured records.

    Args:
        repo_full_name: e.g. "pallets/flask"
        github_client:  Optional pre-built Github instance (useful for testing).
    """

    def __init__(
        self,
        repo_full_name: str,
        github_client: Github | None = None,
    ) -> None:
        self.repo_full_name = repo_full_name
        self._client: Github = github_client or get_github_client()
        self._repo: Repository.Repository | None = None

    def _get_repo(self) -> Repository.Repository:
        if self._repo is None:
            self._repo = safe_github_call(self._client.get_repo, self.repo_full_name)
        return self._repo

    def collect(
        self,
        max_commits: int | None = None,
        branch: str = "main",
    ) -> list[dict[str, Any]]:
        """
        Walk the commit history and return a list of commit records.

        Args:
            max_commits: Maximum number of commits to collect.
                         Defaults to config value (500).
            branch:      Branch to walk. Falls back to the repo default branch.

        Returns:
            List of dicts, one per commit that changed at least one Python file.
        """
        limit = max_commits or cfg["data_collection"]["max_commits_per_repo"]
        min_files = cfg["data_collection"]["min_changed_files"]
        max_files = cfg["data_collection"]["max_changed_files"]

        repo = self._get_repo()

        try:
            commits_paginated = safe_github_call(repo.get_commits, sha=branch)
        except GithubException:
            default_branch = repo.default_branch
            logger.warning(
                f"Branch '{branch}' not found in {self.repo_full_name}. "
                f"Falling back to '{default_branch}'."
            )
            commits_paginated = safe_github_call(repo.get_commits, sha=default_branch)

        records: list[dict[str, Any]] = []
        collected = 0
        skipped_no_py = 0
        skipped_too_large = 0

        logger.info(f"Starting collection from {self.repo_full_name} (limit={limit})")

        for commit in commits_paginated:
            if collected >= limit:
                break

            try:
                full_commit = safe_github_call(lambda c=commit: c)

                files = list(full_commit.files)
                py_files = [_extract_file_record(f) for f in files]
                py_files = [f for f in py_files if f is not None]

                if len(py_files) < min_files:
                    skipped_no_py += 1
                    continue

                if len(py_files) > max_files:
                    skipped_too_large += 1
                    continue

                message = full_commit.commit.message or ""
                author_date = full_commit.commit.author.date

                if author_date.tzinfo is None:
                    author_date = author_date.replace(tzinfo=timezone.utc)
                authored_at = author_date.isoformat()

                record: dict[str, Any] = {
                    "repo": self.repo_full_name,
                    "sha":  full_commit.sha,
                    "parent_sha": (
                        full_commit.parents[0].sha if full_commit.parents else None
                    ),
                    "authored_at": authored_at,
                    "author_name": full_commit.commit.author.name,
                    "message": message[:500],
                    "total_additions": full_commit.stats.additions,
                    "total_deletions": full_commit.stats.deletions,
                    "total_changes":   full_commit.stats.total,
                    "py_files_changed": len(py_files),
                    "files": py_files,
                    **_classify_commit_message(message),
                }

                records.append(record)
                collected += 1

                if collected % 50 == 0:
                    logger.info(
                        f"  {self.repo_full_name}: {collected}/{limit} commits collected"
                    )

            except GithubException as exc:
                logger.warning(f"Skipping commit {commit.sha[:7]}: {exc.status}")
                continue
            except Exception as exc:
                logger.warning(f"Unexpected error on {commit.sha[:7]}: {exc}")
                continue

        logger.info(
            f"Collection complete for {self.repo_full_name}. "
            f"Collected={collected}, "
            f"Skipped(no-py)={skipped_no_py}, "
            f"Skipped(too-large)={skipped_too_large}"
        )
        return records