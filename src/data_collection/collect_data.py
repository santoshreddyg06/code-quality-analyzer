r"""
src/data_collection/collect_data.py
=====================================
Main runner script for Phase 2 data collection.

Run from the project root with:
    .venv\Scripts\python -m src.data_collection.collect_data

What this does:
1. Connects to GitHub using your token
2. Collects commits from each target repository
3. Saves each repo's commits as a JSON file in data/raw/
4. Prints a summary of what was collected

This script is safe to re-run; existing files are NOT overwritten.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.config import cfg, settings
from src.data_collection.commit_collector import CommitCollector
from src.data_collection.github_client import get_github_client, rate_limit_remaining


# --------------------------------------------------------------------------
# Configure loguru to write to both console and a log file
# --------------------------------------------------------------------------
LOG_FILE = Path("data/raw/collection.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
logger.add(LOG_FILE, level="DEBUG", rotation="10 MB")


def save_records(records: list[dict], repo_name: str, output_dir: Path) -> Path:
    """Save records as a JSON file.  Returns the path to the saved file."""
    safe_name = repo_name.replace("/", "__")
    output_path = output_dir / f"{safe_name}_commits.json"

    if output_path.exists():
        logger.warning(f"File already exists, skipping: {output_path.name}")
        return output_path

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "repo": repo_name,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "commit_count": len(records),
                "commits": records,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved {len(records)} commits -> {output_path.name}")
    return output_path


def main() -> None:
    settings.ensure_data_dirs()
    output_dir = settings.data_raw_dir

    target_repos: list[str] = cfg["data_collection"]["target_repos"]
    max_commits: int = cfg["data_collection"]["max_commits_per_repo"]

    client = get_github_client()
    remaining = rate_limit_remaining(client)
    logger.info(f"GitHub API rate limit: {remaining} requests remaining")

    if remaining < 200:
        logger.error(
            f"Only {remaining} API requests remaining. "
            "Please wait for the rate limit to reset (up to 1 hour) before collecting."
        )
        sys.exit(1)

    total_commits = 0
    summary: list[dict] = []

    for repo_name in target_repos:
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting: {repo_name}")
        logger.info(f"{'='*60}")

        safe_name = repo_name.replace("/", "__")
        output_path = output_dir / f"{safe_name}_commits.json"
        if output_path.exists():
            try:
                import json
                with output_path.open("r", encoding="utf-8") as fh:
                    existing_data = json.load(fh)
                count = len(existing_data.get("commits", []))
                logger.info(f"Cached file found with {count} commits: {output_path.name}. Skipping download.")
                summary.append({
                    "repo": repo_name,
                    "commits_collected": count,
                    "output_file": output_path.name,
                    "status": "OK (cached)",
                })
                total_commits += count
                continue
            except Exception as e:
                logger.warning(f"Could not read existing file: {e}")

        try:
            collector = CommitCollector(
                repo_full_name=repo_name,
                github_client=client,
            )
            records = collector.collect(max_commits=max_commits)
            output_path = save_records(records, repo_name, output_dir)

            summary.append({
                "repo": repo_name,
                "commits_collected": len(records),
                "output_file": output_path.name,
                "status": "OK",
            })
            total_commits += len(records)

        except Exception as exc:
            logger.error(f"Failed to collect from {repo_name}: {exc}")
            summary.append({
                "repo": repo_name,
                "commits_collected": 0,
                "output_file": None,
                "status": f"ERROR: {exc}",
            })

    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info("DATA COLLECTION COMPLETE")
    logger.info(f"{'='*60}")
    for s in summary:
        status_icon = "OK" if s["status"] == "OK" else "FAIL"
        logger.info(
            f"  [{status_icon}] {s['repo']:<35} {s['commits_collected']:>4} commits"
        )
    logger.info(f"\n  Total commits collected: {total_commits}")
    remaining_after = rate_limit_remaining(client)
    logger.info(f"  API requests used: {remaining - remaining_after}")
    logger.info(f"  API requests remaining: {remaining_after}")
    client.close()


if __name__ == "__main__":
    main()
