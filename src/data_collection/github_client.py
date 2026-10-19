# -*- coding: utf-8 -*-
"""
src/data_collection/github_client.py
=====================================
Thin wrapper around PyGithub that handles:
  - Authentication via token
  - Automatic rate-limit back-off (waits and retries instead of crashing)
  - Centralised error logging

Usage:
    from src.data_collection.github_client import get_github_client
    g = get_github_client()
    repo = g.get_repo("pallets/flask")
"""

from __future__ import annotations

import time
from loguru import logger
from github import Auth, Github, GithubException, RateLimitExceededException

from src.config import settings


def get_github_client() -> Github:
    """
    Return an authenticated PyGithub client.
    Raises EnvironmentError if GITHUB_TOKEN is missing.
    """
    token = settings.github_token
    auth = Auth.Token(token)
    client = Github(auth=auth, per_page=100)
    logger.debug("GitHub client created.")
    return client


def rate_limit_remaining(client: Github) -> int:
    """Return the number of API requests left in the current window."""
    rl = client.get_rate_limit()
    return rl.rate.remaining


def safe_github_call(func, *args, max_retries: int = 3, **kwargs):
    """
    Call a PyGithub function and automatically retry on rate-limit errors.

    Simple explanation:
        GitHub allows ~5000 API requests per hour.
        If we use them all up, GitHub tells us to wait.
        This function checks for that signal and waits before retrying.

    Args:
        func: Any callable that makes a GitHub API call.
        max_retries: How many times to retry before giving up.

    Returns:
        Whatever func(*args, **kwargs) returns.

    Raises:
        GithubException: For non-rate-limit API errors.
        RuntimeError: If all retries are exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except RateLimitExceededException as exc:
            reset_time = exc.headers.get("x-ratelimit-reset")
            if reset_time:
                wait_seconds = max(int(reset_time) - int(time.time()) + 5, 60)
            else:
                wait_seconds = 60 * attempt
            logger.warning(
                f"Rate limit hit (attempt {attempt}/{max_retries}). "
                f"Waiting {wait_seconds}s before retry..."
            )
            time.sleep(wait_seconds)
        except GithubException as exc:
            logger.error(f"GitHub API error: {exc.status} - {exc.data}")
            raise
    raise RuntimeError(f"GitHub call failed after {max_retries} retries.")