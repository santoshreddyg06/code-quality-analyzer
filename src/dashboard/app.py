"""
src/dashboard/app.py
====================
Interactive Streamlit Dashboard for the AI Code Change Quality Analyzer.
Provides:
1. Live GitHub Commit / Pull Request Analysis
2. Before-and-After Code Editor Analysis
3. Git Diff / Patch Analysis
4. Model Performance & SHAP Feature Insights
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from github import Auth, Github
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env
for p in [PROJECT_ROOT / ".env", Path.cwd() / ".env"]:
    if p.exists():
        load_dotenv(p, override=True)

from src.collector.github_fetcher import GitHubChangeFetcher
from src.explainability.shap_explainer import FEATURE_DESCRIPTIONS, ModelExplainer
from src.features.feature_extractor import extract_features_from_files
from src.metrics.code_metrics import extract_code_metrics
from src.metrics.delta_metrics import compute_delta_metrics
from src.metrics.diff_metrics import extract_diff_metrics, reconstruct_code_versions_from_patch
from src.models.predictor import QualityPredictor

# Page configuration
st.set_page_config(
    page_title="AI Code Change Quality Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .badge-improvement {
        background-color: #d4edda;
        color: #155724;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-degradation {
        background-color: #f8d7da;
        color: #721c24;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def get_env_token() -> str:
    for p in [PROJECT_ROOT / ".env", Path.cwd() / ".env"]:
        if p.exists():
            load_dotenv(p, override=True)
            tok = os.getenv("GITHUB_TOKEN", "").strip()
            if tok:
                return tok
    return os.getenv("GITHUB_TOKEN", "").strip()


def get_github_fetcher(token: str | None = None) -> tuple[GitHubChangeFetcher | None, str | None]:
    try:
        active_token = (token or get_env_token()).strip()
        if not active_token:
            return None, "GITHUB_TOKEN is missing. Please enter your token in the sidebar or in your .env file."
        auth = Auth.Token(active_token)
        client = Github(auth=auth, per_page=100)
        return GitHubChangeFetcher(github_client=client), None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource
def load_ml_services():
    try:
        pred = QualityPredictor()
    except Exception:
        pred = None

    try:
        exp = ModelExplainer()
    except Exception:
        exp = None

    return pred, exp


predictor, explainer = load_ml_services()


# --------------------------------------------------------------------------
# UI Helpers
# --------------------------------------------------------------------------
def render_prediction_badge(pred_label: str, prob_deg: float):
    if "Improvement" in pred_label:
        st.markdown(
            f'<div class="badge-improvement">✅ Prediction: {pred_label} (Healthy)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="badge-degradation">⚠️ Prediction: {pred_label} (Risk Detected)</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Degradation Probability", f"{prob_deg * 100:.1f}%")
    with col2:
        st.metric("Improvement Probability", f"{(1.0 - prob_deg) * 100:.1f}%")


def render_shap_chart(contributions: list[dict[str, Any]]):
    top_items = contributions[:10]
    df = pd.DataFrame(top_items)
    df["color"] = df["shap_value"].apply(lambda v: "#e63946" if v > 0 else "#2a9d8f")

    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=df["display_name"],
        orientation="h",
        marker_color=df["color"],
        text=[f"{v:+.3f}" for v in df["shap_value"]],
        textposition="auto",
    ))
    fig.update_layout(
        title="Top SHAP Feature Influences on Maintainability Risk",
        xaxis_title="SHAP Value (Positive = Increases Risk, Negative = Supports Health)",
        yaxis=dict(autorange="reversed"),
        height=400,
        margin=dict(l=10, r=10, t=40, b=30),
    )
    st.plotly_chart(fig, width="stretch")


def render_metrics_cards(old_m: dict[str, Any], new_m: dict[str, Any], delta_m: dict[str, Any]):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Maintainability Index",
            f"{new_m.get('maintainability_index', 0):.1f}",
            f"{delta_m.get('delta_maintainability_index', 0):+.1f}",
        )
    with c2:
        st.metric(
            "Cyclomatic Complexity",
            f"{new_m.get('total_complexity', 0)}",
            f"{delta_m.get('delta_total_complexity', 0):+d}",
            delta_color="inverse",
        )
    with c3:
        st.metric(
            "Max Nesting Depth",
            f"{new_m.get('max_nesting_depth', 0)}",
            f"{delta_m.get('delta_max_nesting_depth', 0):+d}",
            delta_color="inverse",
        )
    with c4:
        st.metric(
            "Source Lines (SLOC)",
            f"{new_m.get('sloc', 0)}",
            f"{delta_m.get('delta_sloc', 0):+d}",
        )


# --------------------------------------------------------------------------
# Sidebar Navigation & Authentication
# --------------------------------------------------------------------------
st.sidebar.title("🛡️ Code Quality AI")
st.sidebar.markdown("Predict and explain code maintainability impact.")

initial_token = get_env_token()
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Authentication")
user_token = st.sidebar.text_input(
    "GitHub Token",
    value=initial_token,
    type="password",
    help="Loaded automatically from .env or paste your Personal Access Token here.",
)
if user_token:
    os.environ["GITHUB_TOKEN"] = user_token.strip()
    st.sidebar.success("GitHub Token Active ✅")
else:
    st.sidebar.warning("No GitHub Token detected ⚠️")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Choose Analysis Mode:",
    [
        "GitHub Commit / PR",
        "Direct Code Compare",
        "Git Diff / Patch",
        "Model Performance",
    ],
)


# --------------------------------------------------------------------------
# Mode 1: GitHub Commit / PR
# --------------------------------------------------------------------------
if mode == "GitHub Commit / PR":
    st.header("🔍 GitHub Commit & Pull Request Quality Analyzer")
    st.markdown(
        "Enter any public GitHub repository and a commit SHA or PR number to analyze."
    )

    col_repo, col_type, col_id = st.columns([2, 1, 2])
    with col_repo:
        repo_input = st.text_input("Repository (owner/repo)", value="pallets/flask")
    with col_type:
        target_type = st.selectbox("Target", ["Commit SHA", "Pull Request #"])
    with col_id:
        if target_type == "Commit SHA":
            target_id = st.text_input("Commit SHA", value="HEAD~1")
        else:
            target_id = st.text_input("Pull Request #", value="5000")

    if st.button("🚀 Analyze GitHub Change", type="primary"):
        active_fetcher, fetcher_err = get_github_fetcher(token=user_token)
        if not active_fetcher:
            st.error(f"GitHub client is not configured: {fetcher_err}")
        elif not predictor:
            st.error("Model is not trained yet. Run the training pipeline first.")
        else:
            with st.spinner("Fetching changes and computing metrics..."):
                try:
                    repo_clean = repo_input.strip()
                    target_clean = target_id.strip()
                    if target_type == "Commit SHA":
                        change_data = active_fetcher.fetch_commit(repo_clean, target_clean)
                    else:
                        change_data = active_fetcher.fetch_pull_request(repo_clean, int(target_clean))

                    py_files = change_data.get("files", [])
                    if not py_files:
                        sha_short = change_data.get("sha", "")[:7] if change_data.get("sha") else target_clean
                        st.info(
                            f"Commit `{sha_short}` was retrieved successfully, but no Python (.py) files were modified in this change "
                            f"(only documentation, markdown, or configs were touched). Try a commit that touches Python files, such as `42dc739`!"
                        )
                    else:
                        st.success(f"Retrieved {len(py_files)} modified Python file(s).")
                        features = extract_features_from_files(py_files)
                        pred = predictor.predict(features)

                        render_prediction_badge(pred["prediction"], pred["degradation_probability"])

                        # Metadata expander
                        with st.expander("Commit / PR Metadata", expanded=True):
                            st.write(f"**Author:** {change_data.get('author')}")
                            st.write(f"**Date:** {change_data.get('date') or change_data.get('created_at')}")
                            msg = change_data.get('message') or change_data.get('title')
                            st.write(f"**Message:** {msg}")

                        # SHAP Explanation
                        if explainer:
                            exp = explainer.explain_prediction(features)
                            st.subheader("💡 SHAP Decision Factors")
                            render_shap_chart(exp["contributions"])
                            st.markdown(exp["narrative"])

                        # Changed Files
                        st.subheader("📁 Changed Python Files")
                        files_df = pd.DataFrame([
                            {
                                "Filename": f["filename"],
                                "Status": f["status"],
                                "Additions": f["additions"],
                                "Deletions": f["deletions"],
                            }
                            for f in py_files
                        ])
                        st.dataframe(files_df, width="stretch")

                except Exception as exc:
                    st.error(f"Failed to analyze GitHub change: {exc}")


# --------------------------------------------------------------------------
# Mode 2: Direct Code Compare (Before vs After)
# --------------------------------------------------------------------------
elif mode == "Direct Code Compare":
    st.header("📝 Direct Code Comparison Analyzer")
    st.markdown("Paste original code on the left and modified code on the right.")

    # Preset selection
    preset = st.selectbox(
        "Load Sample Scenario:",
        [
            "Custom Code",
            "Example 1: Safe Refactoring (Improves Maintainability)",
            "Example 2: Complex Nested Logic (Potential Degradation)",
        ],
    )

    if preset == "Example 1: Safe Refactoring (Improves Maintainability)":
        default_old = """def process_data(items):
    out = []
    for item in items:
        if item is not None:
            if item.is_valid():
                out.append(item.value)
    return out"""
        default_new = """def process_data(items: list) -> list:
    \"\"\"Extract valid item values cleanly.\"\"\"
    return [item.value for item in items if item and item.is_valid()]"""
    elif preset == "Example 2: Complex Nested Logic (Potential Degradation)":
        default_old = """def handle_request(req):
    return {"status": "ok"}"""
        default_new = """def handle_request(req):
    if req.method == 'POST':
        if req.user:
            if req.user.is_admin:
                for perm in req.user.permissions:
                    if perm == 'CAN_WRITE':
                        try:
                            if req.payload:
                                if len(req.payload) > 100:
                                    return {"error": "payload too large"}
                        except Exception as e:
                            pass
    return {"status": "ok"}"""
    else:
        default_old = "# Paste original code here\ndef compute(x):\n    return x * 2\n"
        default_new = "# Paste modified code here\ndef compute(x):\n    return x * 2 + 1\n"

    col_old, col_new = st.columns(2)
    with col_old:
        old_code = st.text_area("Original Code (Before)", value=default_old, height=220)
    with col_new:
        new_code = st.text_area("Modified Code (After)", value=default_new, height=220)

    if st.button("⚡ Analyze Code Quality Shift", type="primary"):
        if not predictor:
            st.error("Model is not trained yet. Run the training pipeline first.")
        else:
            file_change = {"filename": "editor.py", "old_code": old_code, "new_code": new_code}
            features = extract_features_from_files([file_change])

            old_m = extract_code_metrics(old_code)
            new_m = extract_code_metrics(new_code)
            delta_m = compute_delta_metrics(old_m, new_m)

            pred = predictor.predict(features)
            render_prediction_badge(pred["prediction"], pred["degradation_probability"])

            st.subheader("📊 Metric Shifts")
            render_metrics_cards(old_m.to_dict(), new_m.to_dict(), delta_m.to_dict())

            if explainer:
                exp = explainer.explain_prediction(features)
                st.subheader("💡 Why Did the Model Make This Prediction?")
                render_shap_chart(exp["contributions"])
                st.markdown(exp["narrative"])


# --------------------------------------------------------------------------
# Mode 3: Git Diff / Patch
# --------------------------------------------------------------------------
elif mode == "Git Diff / Patch":
    st.header("🔀 Unified Diff & Patch Quality Analyzer")
    sample_patch = """@@ -1,5 +1,9 @@
 def calculate(a, b):
-    return a + b
+    if b < 0:
+        raise ValueError("b must be positive")
+    # Add safety check
+    return a + b"""

    patch_input = st.text_area("Paste Unified Diff / Patch", value=sample_patch, height=200)

    if st.button("🔍 Inspect Patch", type="primary"):
        if not predictor:
            st.error("Model is not trained yet. Run the training pipeline first.")
        else:
            old_code, new_code = reconstruct_code_versions_from_patch(patch_input)
            diff_m = extract_diff_metrics(patch_input)

            file_change = {
                "filename": "patch.py",
                "patch": patch_input,
                "old_code": old_code,
                "new_code": new_code,
            }
            features = extract_features_from_files([file_change])
            old_m = extract_code_metrics(old_code)
            new_m = extract_code_metrics(new_code)
            delta_m = compute_delta_metrics(old_m, new_m)

            pred = predictor.predict(features)
            render_prediction_badge(pred["prediction"], pred["degradation_probability"])

            st.subheader("📊 Diff Metrics")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Lines Added", f"+{diff_m.lines_added}")
            with c2:
                st.metric("Lines Deleted", f"-{diff_m.lines_deleted}")
            with c3:
                st.metric("Net Lines", f"{diff_m.net_lines:+d}")
            with c4:
                st.metric("Net Branch Keywords", f"{diff_m.net_branch_keywords:+d}")

            if explainer:
                exp = explainer.explain_prediction(features)
                st.subheader("💡 SHAP Explanation")
                render_shap_chart(exp["contributions"])
                st.markdown(exp["narrative"])


# --------------------------------------------------------------------------
# Mode 4: Model Performance
# --------------------------------------------------------------------------
elif mode == "Model Performance":
    st.header("📈 Model Benchmarking & Performance Report")
    metrics_path = Path("models/metrics.json")
    if not metrics_path.exists():
        st.info("Model metrics not found. Train the model first using 'src.models.trainer'.")
    else:
        with metrics_path.open("r", encoding="utf-8") as fh:
            report = json.load(fh)

        st.subheader(f"Winning Model: {report.get('best_model_name')}")

        test_m = report.get("test_metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Test Accuracy", f"{test_m.get('accuracy', 0)*100:.1f}%")
        with c2:
            st.metric("Test Precision", f"{test_m.get('precision', 0)*100:.1f}%")
        with c3:
            st.metric("Test Recall", f"{test_m.get('recall', 0)*100:.1f}%")
        with c4:
            st.metric("Test F1-Score", f"{test_m.get('f1', 0):.3f}")
        with c5:
            st.metric("Test ROC-AUC", f"{test_m.get('roc_auc', 0):.3f}")

        # Validation comparison
        st.subheader("Candidate Model Comparison (Validation Set)")
        val_data = report.get("candidate_val_metrics", {})
        if val_data:
            val_df = pd.DataFrame(val_data).T
            st.dataframe(val_df, width="stretch")
