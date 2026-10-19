"""
tests/test_data_collection.py
==============================
Unit tests for the data collection module.

These tests use mocking so they do NOT make real GitHub API calls.
This keeps tests fast and free of rate-limit concerns.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.data_collection.commit_collector import (
    CommitCollector,
    _classify_commit_message,
    _extract_file_record,
)


# ---------------------------------------------------------------------------
# Tests for _classify_commit_message
# ---------------------------------------------------------------------------
class TestClassifyCommitMessage:
    def test_bug_fix_detected(self):
        result = _classify_commit_message("fix: null pointer in login flow")
        assert result["msg_is_bugfix"] is True
        assert result["msg_is_revert"] is False

    def test_revert_detected(self):
        result = _classify_commit_message("Revert 'Add new feature'")
        assert result["msg_is_revert"] is True
        assert result["msg_is_bugfix"] is False

    def test_test_commit_detected(self):
        result = _classify_commit_message("Add unittest for auth module")
        assert result["msg_is_test"] is True

    def test_docs_commit_detected(self):
        result = _classify_commit_message("Update README with new examples")
        assert result["msg_is_docs"] is True

    def test_feature_commit_no_flags(self):
        result = _classify_commit_message("Add support for async handlers")
        assert result["msg_is_bugfix"] is False
        assert result["msg_is_revert"] is False
        assert result["msg_is_test"] is False
        assert result["msg_is_docs"] is False

    def test_empty_message(self):
        result = _classify_commit_message("")
        assert all(v is False for v in result.values())


# ---------------------------------------------------------------------------
# Tests for _extract_file_record
# ---------------------------------------------------------------------------
class TestExtractFileRecord:
    def _make_file(self, filename, status="modified", additions=5,
                   deletions=2, changes=7, patch="@@ diff @@"):
        f = MagicMock()
        f.filename = filename
        f.status = status
        f.additions = additions
        f.deletions = deletions
        f.changes = changes
        f.patch = patch
        return f

    def test_python_file_extracted(self):
        f = self._make_file("src/app.py")
        result = _extract_file_record(f)
        assert result is not None
        assert result["filename"] == "src/app.py"
        assert result["additions"] == 5
        assert result["deletions"] == 2

    def test_non_python_file_skipped(self):
        for name in ["README.md", "config.yaml", "Makefile", "styles.css"]:
            f = self._make_file(name)
            assert _extract_file_record(f) is None, f"Should skip {name}"

    def test_file_with_no_patch(self):
        f = self._make_file("utils.py", patch=None)
        result = _extract_file_record(f)
        assert result is not None
        assert result["patch"] is None


# ---------------------------------------------------------------------------
# Tests for CommitCollector (mocked GitHub client)
# ---------------------------------------------------------------------------
class TestCommitCollector:
    def _make_mock_commit(self, sha="abc123", message="fix: bug", py_file=True):
        """Build a mock GitHub commit object."""
        commit = MagicMock()
        commit.sha = sha

        # Parent
        parent = MagicMock()
        parent.sha = "parent" + sha
        commit.parents = [parent]

        # Commit metadata
        commit.commit.message = message
        from datetime import datetime, timezone
        commit.commit.author.date = datetime(2023, 6, 1, tzinfo=timezone.utc)
        commit.commit.author.name = "Test Author"

        # Stats
        commit.stats.additions = 10
        commit.stats.deletions = 5
        commit.stats.total = 15

        # Files
        f = MagicMock()
        f.filename = "src/app.py" if py_file else "README.md"
        f.status = "modified"
        f.additions = 10
        f.deletions = 5
        f.changes = 15
        f.patch = "@@ -1,5 +1,10 @@\n some code"
        commit.files = [f]

        return commit

    def test_collect_returns_records_for_python_commits(self):
        mock_client = MagicMock()
        mock_repo = MagicMock()
        mock_repo.default_branch = "main"

        commits = [self._make_mock_commit(sha=f"sha{i:03d}") for i in range(5)]
        mock_repo.get_commits.return_value = iter(commits)
        mock_client.get_repo.return_value = mock_repo

        collector = CommitCollector("test/repo", github_client=mock_client)
        records = collector.collect(max_commits=5)

        assert len(records) == 5
        assert records[0]["repo"] == "test/repo"
        assert records[0]["py_files_changed"] == 1
        assert "msg_is_bugfix" in records[0]

    def test_collect_skips_non_python_commits(self):
        mock_client = MagicMock()
        mock_repo = MagicMock()
        mock_repo.default_branch = "main"

        # Mix: 2 Python commits, 2 non-Python commits
        commits = [
            self._make_mock_commit(sha="py1", py_file=True),
            self._make_mock_commit(sha="nopy1", py_file=False),
            self._make_mock_commit(sha="py2", py_file=True),
            self._make_mock_commit(sha="nopy2", py_file=False),
        ]
        mock_repo.get_commits.return_value = iter(commits)
        mock_client.get_repo.return_value = mock_repo

        collector = CommitCollector("test/repo", github_client=mock_client)
        records = collector.collect(max_commits=10)

        assert len(records) == 2
        assert all(r["py_files_changed"] >= 1 for r in records)
