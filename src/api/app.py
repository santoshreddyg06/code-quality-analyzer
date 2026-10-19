"""
src/api/app.py
==============
FastAPI REST API for code quality analysis, impact prediction, and SHAP explainability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collector.github_fetcher import GitHubChangeFetcher
from src.explainability.shap_explainer import ModelExplainer
from src.features.feature_extractor import extract_features_from_files
from src.metrics.code_metrics import extract_code_metrics
from src.metrics.delta_metrics import compute_delta_metrics
from src.metrics.diff_metrics import extract_diff_metrics, reconstruct_code_versions_from_patch
from src.models.predictor import QualityPredictor

app = FastAPI(
    title="AI Code Change Quality Analyzer API",
    description="Predicts maintainability impact and explains predictions using SHAP.",
    version="1.0.0",
)

# Enable CORS for local dashboards and frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons for model and explainer
_predictor: QualityPredictor | None = None
_explainer: ModelExplainer | None = None
_fetcher: GitHubChangeFetcher | None = None


def get_predictor() -> QualityPredictor:
    global _predictor
    if _predictor is None:
        _predictor = QualityPredictor()
    return _predictor


def get_explainer() -> ModelExplainer:
    global _explainer
    if _explainer is None:
        _explainer = ModelExplainer()
    return _explainer


def get_fetcher() -> GitHubChangeFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = GitHubChangeFetcher()
    return _fetcher


# --------------------------------------------------------------------------
# Request / Response Schemas
# --------------------------------------------------------------------------
class CodeAnalysisRequest(BaseModel):
    old_code: str = Field(..., description="Original Python source code")
    new_code: str = Field(..., description="Modified Python source code")
    filename: str = Field("snippet.py", description="Relative filename")


class DiffAnalysisRequest(BaseModel):
    patch: str = Field(..., description="Unified diff patch")
    filename: str = Field("snippet.py", description="Relative filename")


class GitHubAnalysisRequest(BaseModel):
    repo: str = Field(..., description="GitHub repository full name, e.g. 'pallets/flask'")
    commit_sha: str | None = Field(None, description="Commit SHA to inspect")
    pr_number: int | None = Field(None, description="Pull Request number to inspect")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health")
def health_check() -> dict[str, Any]:
    model_exists = Path("models/best_model.joblib").exists()
    return {
        "status": "healthy",
        "model_loaded": model_exists,
        "version": "1.0.0",
    }


@app.get("/api/model/info")
def model_info() -> dict[str, Any]:
    metrics_path = Path("models/metrics.json")
    if not metrics_path.exists():
        return {"status": "Model not trained yet."}
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@app.post("/api/analyze/code")
def analyze_code_change(req: CodeAnalysisRequest) -> dict[str, Any]:
    file_change = {
        "filename": req.filename,
        "old_code": req.old_code,
        "new_code": req.new_code,
    }
    features = extract_features_from_files([file_change])

    old_m = extract_code_metrics(req.old_code)
    new_m = extract_code_metrics(req.new_code)
    delta_m = compute_delta_metrics(old_m, new_m)

    try:
        predictor = get_predictor()
        prediction = predictor.predict(features)
    except FileNotFoundError:
        prediction = {"prediction": "Model not trained", "risk_level": "Unknown"}

    try:
        explainer = get_explainer()
        explanation = explainer.explain_prediction(features)
    except Exception as exc:
        explanation = {"narrative": f"Explanation unavailable: {exc}", "contributions": []}

    return {
        "prediction": prediction,
        "explanation": explanation,
        "metrics": {
            "old": old_m.to_dict(),
            "new": new_m.to_dict(),
            "delta": delta_m.to_dict(),
        },
        "features": features,
    }


@app.post("/api/analyze/diff")
def analyze_diff(req: DiffAnalysisRequest) -> dict[str, Any]:
    old_code, new_code = reconstruct_code_versions_from_patch(req.patch)
    diff_m = extract_diff_metrics(req.patch)

    file_change = {
        "filename": req.filename,
        "patch": req.patch,
        "old_code": old_code,
        "new_code": new_code,
    }
    features = extract_features_from_files([file_change])

    old_m = extract_code_metrics(old_code)
    new_m = extract_code_metrics(new_code)
    delta_m = compute_delta_metrics(old_m, new_m)

    try:
        predictor = get_predictor()
        prediction = predictor.predict(features)
    except FileNotFoundError:
        prediction = {"prediction": "Model not trained", "risk_level": "Unknown"}

    try:
        explainer = get_explainer()
        explanation = explainer.explain_prediction(features)
    except Exception as exc:
        explanation = {"narrative": f"Explanation unavailable: {exc}", "contributions": []}

    return {
        "prediction": prediction,
        "explanation": explanation,
        "metrics": {
            "old": old_m.to_dict(),
            "new": new_m.to_dict(),
            "delta": delta_m.to_dict(),
            "diff": diff_m.to_dict(),
        },
        "features": features,
    }


@app.post("/api/analyze/github")
def analyze_github(req: GitHubAnalysisRequest) -> dict[str, Any]:
    fetcher = get_fetcher()

    if req.commit_sha:
        change_data = fetcher.fetch_commit(req.repo, req.commit_sha)
    elif req.pr_number:
        change_data = fetcher.fetch_pull_request(req.repo, req.pr_number)
    else:
        raise HTTPException(status_code=400, detail="Must provide either commit_sha or pr_number")

    py_files = change_data.get("files", [])
    if not py_files:
        return {
            "metadata": change_data,
            "message": "No Python files were modified in this change.",
            "prediction": {"prediction": "No Python Changes", "risk_level": "None"},
        }

    features = extract_features_from_files(py_files)

    try:
        predictor = get_predictor()
        prediction = predictor.predict(features)
    except FileNotFoundError:
        prediction = {"prediction": "Model not trained", "risk_level": "Unknown"}

    try:
        explainer = get_explainer()
        explanation = explainer.explain_prediction(features)
    except Exception as exc:
        explanation = {"narrative": f"Explanation unavailable: {exc}", "contributions": []}

    return {
        "metadata": {
            "type": change_data.get("type"),
            "repo": change_data.get("repo"),
            "sha": change_data.get("sha"),
            "pr_number": change_data.get("pr_number"),
            "author": change_data.get("author"),
            "date": change_data.get("date") or change_data.get("created_at"),
            "message": change_data.get("message") or change_data.get("title"),
            "stats": change_data.get("stats"),
            "files_count": len(py_files),
        },
        "prediction": prediction,
        "explanation": explanation,
        "features": features,
        "files": [
            {
                "filename": f["filename"],
                "status": f["status"],
                "additions": f["additions"],
                "deletions": f["deletions"],
            }
            for f in py_files
        ],
    }
