"""
src/config.py
=============
Central configuration loader.

Reads:
  - configs/config.yaml  (project settings, safe to commit)
  - .env                  (secrets such as GITHUB_TOKEN, never committed)

Usage:
    from src.config import cfg, settings
    token = settings.github_token
    max_commits = cfg["data_collection"]["max_commits_per_repo"]
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate project root (the directory that contains this src/ package)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Load .env  (no error if file is absent – env vars may be set externally)
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Load YAML config
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


cfg: dict = _load_yaml(PROJECT_ROOT / "configs" / "config.yaml")


# ---------------------------------------------------------------------------
# Typed settings (secrets and derived paths)
# ---------------------------------------------------------------------------
class Settings:
    """
    Strongly typed wrapper for environment variables.
    Raises clear errors when required values are missing.
    """

    @property
    def github_token(self) -> str:
        token = os.getenv("GITHUB_TOKEN", "")
        if not token:
            raise EnvironmentError(
                "GITHUB_TOKEN is not set. "
                "Copy .env.example to .env and add your GitHub Personal Access Token."
            )
        return token

    @property
    def api_base_url(self) -> str:
        return os.getenv("API_BASE_URL", "http://localhost:8000")

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO").upper()

    # Convenience path properties
    @property
    def data_raw_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "models"

    def ensure_data_dirs(self) -> None:
        """Create data directories if they do not exist."""
        for d in (self.data_raw_dir, self.data_processed_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
