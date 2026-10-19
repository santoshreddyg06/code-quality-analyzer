"""
src/collector/github_fetcher.py
===============================
Retrieves code changes, diffs, and file versions for any specified
GitHub repository commit or pull request.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from github import Github, Repository
from github.GithubException import GithubException

from src.data_collection.github_client import get_github_client, safe_github_call


class GitHubChangeFetcher:
    """Fetches commit or pull request changes for quality analysis."""

    def __init__(self, github_client: Github | None = None) -> None:
        self.client = github_client or get_github_client()

    def fetch_commit(self, repo_full_name: str, commit_sha: str) -> dict[str, Any]:
        """
        Fetch details and file changes for a specific commit.

        Args:
            repo_full_name: e.g. "pallets/flask"
            commit_sha: Full or short commit SHA.

        Returns:
            Dict containing metadata and a list of changed Python files.
        """
        logger.info(f"Fetching commit {commit_sha} from {repo_full_name}")
        repo = safe_github_call(self.client.get_repo, repo_full_name)
        commit = safe_github_call(repo.get_commit, commit_sha)

        file_changes = []
        for f in commit.files:
            if not f.filename.endswith(".py"):
                continue

            file_changes.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": getattr(f, "patch", ""),
                "raw_url": f.raw_url,
            })

        return {
            "type": "commit",
            "repo": repo_full_name,
            "sha": commit.sha,
            "author": commit.commit.author.name,
            "date": commit.commit.author.date.isoformat(),
            "message": commit.commit.message,
            "stats": {
                "total_additions": commit.stats.additions,
                "total_deletions": commit.stats.deletions,
                "total_changes": commit.stats.total,
            },
            "files": file_changes,
        }

    def fetch_pull_request(self, repo_full_name: str, pr_number: int) -> dict[str, Any]:
        """
        Fetch details and file changes for a specific pull request.

        Args:
            repo_full_name: e.g. "pallets/flask"
            pr_number: Pull request integer ID.

        Returns:
            Dict containing PR metadata and changed Python files.
        """
        logger.info(f"Fetching PR #{pr_number} from {repo_full_name}")
        repo = safe_github_call(self.client.get_repo, repo_full_name)
        pr = safe_github_call(repo.get_pull, pr_number)

        file_changes = []
        for f in pr.get_files():
            if not f.filename.endswith(".py"):
                continue

            file_changes.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": getattr(f, "patch", ""),
                "raw_url": f.raw_url,
            })

        return {
            "type": "pull_request",
            "repo": repo_full_name,
            "pr_number": pr_number,
            "title": pr.title,
            "body": pr.body or "",
            "author": pr.user.login if pr.user else "Unknown",
            "state": pr.state,
            "created_at": pr.created_at.isoformat(),
            "stats": {
                "total_additions": pr.additions,
                "total_deletions": pr.deletions,
                "total_changed_files": pr.changed_files,
            },
            "files": file_changes,
        }
