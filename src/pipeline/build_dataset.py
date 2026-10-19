"""
src/pipeline/build_dataset.py
=============================
Transforms collected raw commits from data/raw/*.json into clean tabular datasets
ready for machine learning:
- data/processed/dataset.csv (full dataset)
- data/processed/train.csv
- data/processed/val.csv
- data/processed/test.csv

Run with:
    .venv\Scripts\python -m src.pipeline.build_dataset
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import cfg, settings
from src.features.feature_extractor import FEATURE_COLUMNS, extract_features_from_commit_record
from src.labeling.label_generator import compute_maintainability_label


def process_raw_file(file_path: Path) -> list[dict]:
    """Process a single repository JSON file into a list of feature dicts."""
    logger.info(f"Processing raw file: {file_path.name}")
    with file_path.open("r", encoding="utf-8") as fh:
        raw_data = json.load(fh)

    commits = raw_data.get("commits", [])
    logger.info(f"  Found {len(commits)} commits in {file_path.name}")

    rows: list[dict] = []
    for commit in commits:
        try:
            feat_row = extract_features_from_commit_record(commit)
            label_info = compute_maintainability_label(feat_row)
            row = {
                **feat_row,
                "label": label_info["label"],
                "label_name": label_info["label_name"],
            }
            rows.append(row)
        except Exception as exc:
            logger.warning(f"Error processing commit {commit.get('sha', 'unknown')}: {exc}")
            continue

    logger.info(f"  Successfully extracted features for {len(rows)} commits.")
    return rows


def build_dataset() -> pd.DataFrame:
    """Build the complete dataset from all raw JSON files and split into train/val/test."""
    settings.ensure_data_dirs()
    raw_dir = settings.data_raw_dir
    processed_dir = settings.data_processed_dir

    raw_files = list(raw_dir.glob("*_commits.json"))
    if not raw_files:
        raise FileNotFoundError(f"No *_commits.json files found in {raw_dir}")

    all_rows: list[dict] = []
    for raw_file in raw_files:
        rows = process_raw_file(raw_file)
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError("No valid commit records could be processed.")

    df = pd.DataFrame(all_rows)
    logger.info(f"\nTotal dataset shape: {df.shape}")
    logger.info(f"Class distribution:\n{df['label_name'].value_counts(normalize=True).round(3)}")

    # Save full dataset
    full_path = processed_dir / "dataset.csv"
    df.to_csv(full_path, index=False)
    logger.info(f"Saved full dataset -> {full_path}")

    # Train / Val / Test split
    test_size = cfg["models"].get("test_size", 0.15)
    val_size = cfg["models"].get("val_size", 0.15)
    random_state = cfg["models"].get("random_state", 42)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val_df["label"],
    )

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    logger.info(f"Saved splits -> {processed_dir}")
    return df


if __name__ == "__main__":
    build_dataset()
