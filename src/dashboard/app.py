"""
src/dashboard/app.py
====================
Production-grade developer dashboard for the AI Code Change Quality Analyzer.
Crafted with high data density, minimal marketing fluff, and developer-first aesthetics.
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
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
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

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Code Quality Guard | Engineering Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Clean Developer Aesthetics (Linear / Vercel / GitHub Theme)
# --------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #18181b;
    }

    code, pre, .stCodeBlock, [data-testid="stMarkdownContainer"] code {
        font-family: 'JetBrains Mono', ui-monospace, monospace !important;
        font-size: 0.88em;
    }

    /* Minimalist Header Strip */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 20px;
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #09090b;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .brand-tag {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        background: #f4f4f5;
        border: 1px solid #e4e4e7;
        color: #71717a;
        font-family: 'JetBrains Mono', monospace;
    }

    .system-status {
        display: flex;
        align-items: center;
        gap: 16px;
        font-size: 0.8rem;
        color: #71717a;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        margin-right: 6px;
    }

    /* Metric Cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }

    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #71717a;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }

    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #09090b;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }

    .stat-delta {
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    .stat-delta.positive { color: #10b981; }
    .stat-delta.negative { color: #ef4444; }
    .stat-delta.neutral { color: #71717a; }

    /* Verdict Card */
    .verdict-banner-safe {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 16px 0;
    }

    .verdict-banner-risk {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 16px 0;
    }

    .verdict-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .verdict-safe-text { color: #166534; }
    .verdict-risk-text { color: #991b1b; }

    .verdict-meta {
        font-size: 0.85rem;
        color: #52525b;
        display: flex;
        gap: 20px;
        margin-top: 8px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Preset Buttons */
    .preset-chip {
        font-size: 0.8rem;
        color: #3f3f46;
        border: 1px solid #e4e4e7;
        border-radius: 6px;
        padding: 4px 10px;
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Backend Token & Service Loaders
# --------------------------------------------------------------------------
def get_server_token() -> str:
    try:
        if "GITHUB_TOKEN" in st.secrets:
            tok = str(st.secrets["GITHUB_TOKEN"]).strip()
            if tok:
                return tok
    except Exception:
        pass

    for p in [PROJECT_ROOT / ".env", Path.cwd() / ".env"]:
        if p.exists():
            load_dotenv(p, override=True)
            tok = os.getenv("GITHUB_TOKEN", "").strip()
            if tok:
                return tok

    return os.getenv("GITHUB_TOKEN", "").strip()


def get_github_fetcher(token: str | None = None) -> tuple[GitHubChangeFetcher | None, str | None]:
    try:
        active_token = (token.strip() if token and token.strip() else get_server_token())
        if not active_token:
            return None, "No GITHUB_TOKEN configured. Provide one in the sidebar or set GITHUB_TOKEN on the server."
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
# UI Helpers & Visualizations
# --------------------------------------------------------------------------
def render_navbar(token_configured: bool):
    tok_badge = '<span style="color:#10b981;">● Active (Secure)</span>' if token_configured else '<span style="color:#f59e0b;">● Unset</span>'
    st.markdown(f"""
    <div class="top-navbar">
        <div class="brand-title">
            <span>Code Quality Guard</span>
            <span class="brand-tag">v1.2.0</span>
            <span class="brand-tag">XGBoost Engine</span>
        </div>
        <div class="system-status">
            <span><span class="status-dot"></span>Inference Ready</span>
            <span>GitHub Auth: {tok_badge}</span>
            <span class="brand-tag">25 Delta Metrics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_verdict(pred_label: str, prob_deg: float, risk_level: str):
    is_safe = "Improvement" in pred_label or prob_deg < 0.50
    prob_safe = 1.0 - prob_deg

    if is_safe:
        st.markdown(f"""
        <div class="verdict-banner-safe">
            <div class="verdict-title verdict-safe-text">
                ✓ No Degradation Detected • Likely Safe Refactor
            </div>
            <div style="font-size:0.88rem; color:#15803d;">
                The code diff preserves or improves maintainability. Complexity remains within bounds and documentation density is healthy.
            </div>
            <div class="verdict-meta">
                <span>Model Confidence: <b>{prob_safe * 100:.1f}%</b></span>
                <span>Degradation Risk: <b>{prob_deg * 100:.1f}%</b></span>
                <span>Risk Tier: <b>{risk_level}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-banner-risk">
            <div class="verdict-title verdict-risk-text">
                ▲ Maintainability Degradation Risk Detected
            </div>
            <div style="font-size:0.88rem; color:#b91c1c;">
                Significant increase in cyclomatic complexity, nesting depth, or control flow branching detected in this diff.
            </div>
            <div class="verdict-meta">
                <span>Degradation Probability: <b>{prob_deg * 100:.1f}%</b></span>
                <span>Risk Tier: <b>{risk_level}</b></span>
                <span>Status: <b>Review Recommended</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_stat_box(title: str, value: str, delta: str, delta_class: str = "neutral"):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{title}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-delta {delta_class}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_row(old_m: dict[str, Any], new_m: dict[str, Any], delta_m: dict[str, Any]):
    c1, c2, c3, c4 = st.columns(4)

    # Maintainability Index
    mi_new = new_m.get("maintainability_index", 0.0)
    mi_delta = delta_m.get("delta_maintainability_index", 0.0)
    mi_class = "positive" if mi_delta >= 0 else "negative"
    with c1:
        render_stat_box("Maintainability Index", f"{mi_new:.1f}", f"{mi_delta:+.1f} pts", mi_class)

    # Cyclomatic Complexity
    cc_new = new_m.get("total_complexity", 0)
    cc_delta = delta_m.get("delta_total_complexity", 0)
    cc_class = "positive" if cc_delta <= 0 else "negative"
    with c2:
        render_stat_box("Cyclomatic Complexity", f"{cc_new}", f"{cc_delta:+d} paths", cc_class)

    # Max Nesting
    nest_new = new_m.get("max_nesting_depth", 0)
    nest_delta = delta_m.get("delta_max_nesting_depth", 0)
    nest_class = "positive" if nest_delta <= 0 else "negative"
    with c3:
        render_stat_box("Max Nesting Depth", f"{nest_new}", f"{nest_delta:+d} levels", nest_class)

    # SLOC
    sloc_new = new_m.get("sloc", 0)
    sloc_delta = delta_m.get("delta_sloc", 0)
    with c4:
        render_stat_box("Source Lines (SLOC)", f"{sloc_new}", f"{sloc_delta:+d} lines", "neutral")


def render_shap_figure(contributions: list[dict[str, Any]]):
    top_items = contributions[:10]
    df = pd.DataFrame(top_items)
    df["color"] = df["shap_value"].apply(lambda v: "#ef4444" if v > 0 else "#10b981")

    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=df["display_name"],
        orientation="h",
        marker=dict(
            color=df["color"],
            line=dict(width=0),
        ),
        text=[f"{v:+.3f}" for v in df["shap_value"]],
        textposition="auto",
    ))
    fig.update_layout(
        title=dict(
            text="SHAP Feature Attribution (TreeExplainer)",
            font=dict(size=14, color="#18181b", family="Inter"),
        ),
        xaxis_title="SHAP Value (Positive = Increases Risk, Negative = Supports Health)",
        yaxis=dict(autorange="reversed"),
        height=340,
        margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11),
    )
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# Main Top Navbar
# --------------------------------------------------------------------------
server_token_present = bool(get_server_token())
render_navbar(server_token_present)

# --------------------------------------------------------------------------
# Minimalist Sidebar (Strictly Developer Controls & Auth)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Authentication")
    if server_token_present:
        st.success("Server Token: Configured Securely")
        st.caption("Backend token is active in server memory. You may optionally enter a custom token below to override.")
    else:
        st.warning("No Server Token")
        st.caption("Enter a personal GitHub token to query the GitHub API.")

    custom_token = st.text_input(
        "Custom Token (Optional)",
        value="",
        type="password",
        placeholder="Enter personal token...",
        help="Optional personal token. Overrides server token for this browser session only.",
    )

    st.markdown("---")
    st.markdown("### System Specifications")
    st.markdown("""
    - **Engine:** XGBoost Classifier
    - **Dataset:** 450 mined Python commits
    - **Offline Accuracy:** 98.5%
    - **ROC-AUC:** 0.996
    - **Parser:** Radon, Lizard, Python AST
    - **Explainability:** TreeExplainer SHAP
    """)


# --------------------------------------------------------------------------
# Main Workspace Tabs (GitHub Inspector, Sandbox, Patch, Model, Docs)
# --------------------------------------------------------------------------
tab_github, tab_compare, tab_patch, tab_metrics, tab_docs = st.tabs([
    "GitHub Inspector",
    "Code Sandbox",
    "Patch Inspector",
    "Model Benchmarks",
    "Metric Specifications",
])


# ==========================================================================
# TAB 1: GitHub Commit / PR Inspector
# ==========================================================================
with tab_github:
    st.markdown("#### Inspect Live GitHub Commit or Pull Request")
    st.caption("Fetches unified diffs, reconstructs file ASTs, and calculates code quality deltas.")

    # Inline Quick Presets
    st.markdown("**Quick Presets:**")
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("pallets/flask @ 42dc739", help="Python 3.12 AST refactor"):
            st.session_state["gh_repo"] = "pallets/flask"
            st.session_state["gh_type"] = "Commit SHA"
            st.session_state["gh_target"] = "42dc739"
    with p2:
        if st.button("psf/requests @ c8a4087", help="Bugfix commit in Requests"):
            st.session_state["gh_repo"] = "psf/requests"
            st.session_state["gh_type"] = "Commit SHA"
            st.session_state["gh_target"] = "c8a4087"
    with p3:
        if st.button("pallets/flask @ HEAD~1", help="Recent Flask repository commit"):
            st.session_state["gh_repo"] = "pallets/flask"
            st.session_state["gh_type"] = "Commit SHA"
            st.session_state["gh_target"] = "HEAD~1"

    init_repo = st.session_state.get("gh_repo", "pallets/flask")
    init_type = st.session_state.get("gh_type", "Commit SHA")
    init_target = st.session_state.get("gh_target", "42dc739")

    col_repo, col_type, col_id = st.columns([3, 2, 3])
    with col_repo:
        repo_input = st.text_input("Repository (owner/repo)", value=init_repo)
    with col_type:
        type_options = ["Commit SHA", "Pull Request #"]
        target_type = st.selectbox("Target Type", type_options, index=0 if init_type == "Commit SHA" else 1)
    with col_id:
        target_id = st.text_input("Commit SHA or PR #", value=init_target)

    if st.button("Analyze Commit / PR", type="primary"):
        active_fetcher, fetcher_err = get_github_fetcher(token=custom_token)
        if not active_fetcher:
            st.error(f"GitHub client error: {fetcher_err}")
        elif not predictor:
            st.error("Predictor model artifact is not loaded.")
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
                        st.info(f"Commit `{sha_short}` retrieved, but contains no modified Python files. Try a commit touching `.py` files such as `42dc739`.")
                    else:
                        features = extract_features_from_files(py_files)
                        pred = predictor.predict(features)

                        render_verdict(pred["prediction"], pred["degradation_probability"], pred.get("risk_level", "Low Risk"))

                        # Changed Files Summary
                        st.markdown(f"**Changed Python Files ({len(py_files)}):**")
                        files_df = pd.DataFrame([
                            {
                                "Filename": f["filename"],
                                "Status": f["status"],
                                "Additions": f"+{f['additions']}",
                                "Deletions": f"-{f['deletions']}",
                            }
                            for f in py_files
                        ])
                        st.dataframe(files_df, width="stretch", hide_index=True)

                        # Metadata Strip
                        with st.expander("Commit Details & Message", expanded=False):
                            st.markdown(f"**Author:** `{change_data.get('author')}` | **Date:** `{change_data.get('date') or change_data.get('created_at')}`")
                            msg = change_data.get('message') or change_data.get('title')
                            st.code(msg, language="text")

                        # SHAP Explanation
                        if explainer:
                            exp = explainer.explain_prediction(features)
                            render_shap_figure(exp["contributions"])
                            st.markdown(exp["narrative"])

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")


# ==========================================================================
# TAB 2: Code Comparison Sandbox (Before vs After)
# ==========================================================================
with tab_compare:
    st.markdown("#### Direct Code Comparison Sandbox")
    st.caption("Compare original and refactored Python snippets side-by-side to evaluate quality shifts.")

    scenario = st.selectbox(
        "Load Code Scenario:",
        [
            "Custom Snippets",
            "Refactoring: Nested Loops → List Comprehension (Improves MI)",
            "Degradation: Deep Nesting & Branching Anti-Pattern (High CC)",
            "Documentation: Adding Docstrings & Type Hints (Boosts Documentation Density)",
        ],
    )

    if scenario == "Refactoring: Nested Loops → List Comprehension (Improves MI)":
        s_old = """def process_items(items):
    output = []
    for item in items:
        if item is not None:
            if item.is_valid():
                output.append(item.value)
    return output"""
        s_new = """def process_items(items: list) -> list:
    \"\"\"Extract valid values concisely using list comprehension.\"\"\"
    return [item.value for item in items if item and item.is_valid()]"""

    elif scenario == "Degradation: Deep Nesting & Branching Anti-Pattern (High CC)":
        s_old = """def handle_request(req):
    return {"status": "ok"}"""
        s_new = """def handle_request(req):
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

    elif scenario == "Documentation: Adding Docstrings & Type Hints (Boosts Documentation Density)":
        s_old = """def calculate_mortgage(principal, rate, years):
    n = years * 12
    r = rate / 12 / 100
    monthly = principal * (r * (1 + r)**n) / ((1 + r)**n - 1)
    return monthly"""
        s_new = """def calculate_mortgage(principal: float, rate: float, years: int) -> float:
    \"\"\"
    Calculate fixed monthly mortgage payments.

    Args:
        principal: Total loan amount borrowed.
        rate: Annual interest rate as a percentage (e.g. 5.5).
        years: Loan duration in years.

    Returns:
        Fixed monthly payment amount.
    \"\"\"
    n = years * 12
    r = rate / 12 / 100
    monthly = principal * (r * (1 + r)**n) / ((1 + r)**n - 1)
    return monthly"""

    else:
        s_old = "def compute(x):\n    return x * 2\n"
        s_new = "def compute(x):\n    return x * 2 + 1\n"

    c_left, c_right = st.columns(2)
    with c_left:
        code_before = st.text_area("Original Code (Before)", value=s_old, height=220)
    with c_right:
        code_after = st.text_area("Modified Code (After)", value=s_new, height=220)

    if st.button("Evaluate Quality Shift", type="primary"):
        if not predictor:
            st.error("Predictor model is not available.")
        else:
            file_change = {"filename": "sandbox.py", "old_code": code_before, "new_code": code_after}
            features = extract_features_from_files([file_change])

            old_m = extract_code_metrics(code_before)
            new_m = extract_code_metrics(code_after)
            delta_m = compute_delta_metrics(old_m, new_m)

            pred = predictor.predict(features)
            render_verdict(pred["prediction"], pred["degradation_probability"], pred.get("risk_level", "Low Risk"))

            st.markdown("##### Metric Deltas")
            render_metrics_row(old_m.to_dict(), new_m.to_dict(), delta_m.to_dict())

            if explainer:
                exp = explainer.explain_prediction(features)
                render_shap_figure(exp["contributions"])
                st.markdown(exp["narrative"])


# ==========================================================================
# TAB 3: Patch Inspector (Unified Diff)
# ==========================================================================
with tab_patch:
    st.markdown("#### Unified Patch Inspector")
    st.caption("Paste standard unified diff chunks from git diff or pull request review tools.")

    default_diff = """@@ -1,5 +1,9 @@
 def calculate(a, b):
-    return a + b
+    if b < 0:
+        raise ValueError("b must be positive")
+    # Add safety check
+    return a + b"""

    patch_text = st.text_area("Unified Git Diff", value=default_diff, height=180)

    if st.button("Inspect Patch Quality", type="primary"):
        if not predictor:
            st.error("Predictor model is not available.")
        else:
            old_code, new_code = reconstruct_code_versions_from_patch(patch_text)
            diff_m = extract_diff_metrics(patch_text)

            file_change = {
                "filename": "patch.py",
                "patch": patch_text,
                "old_code": old_code,
                "new_code": new_code,
            }
            features = extract_features_from_files([file_change])
            old_m = extract_code_metrics(old_code)
            new_m = extract_code_metrics(new_code)
            delta_m = compute_delta_metrics(old_m, new_m)

            pred = predictor.predict(features)
            render_verdict(pred["prediction"], pred["degradation_probability"], pred.get("risk_level", "Low Risk"))

            st.markdown("##### Diff Statistics")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                render_stat_box("Lines Added", f"+{diff_m.lines_added}", "Diff additions", "neutral")
            with d2:
                render_stat_box("Lines Deleted", f"-{diff_m.lines_deleted}", "Diff deletions", "neutral")
            with d3:
                render_stat_box("Net Lines", f"{diff_m.net_lines:+d}", "Net code change", "neutral")
            with d4:
                b_class = "positive" if diff_m.net_branch_keywords <= 0 else "negative"
                render_stat_box("Branch Keywords", f"{diff_m.net_branch_keywords:+d}", "Control flow keywords", b_class)

            if explainer:
                exp = explainer.explain_prediction(features)
                render_shap_figure(exp["contributions"])
                st.markdown(exp["narrative"])


# ==========================================================================
# TAB 4: Model Benchmarks & Metrics
# ==========================================================================
with tab_metrics:
    st.markdown("#### Offline Model Evaluation Benchmarks")
    metrics_path = Path("models/metrics.json")
    if not metrics_path.exists():
        st.info("No metrics file found at models/metrics.json.")
    else:
        with metrics_path.open("r", encoding="utf-8") as fh:
            report = json.load(fh)

        test_m = report.get("test_metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            render_stat_box("Accuracy", f"{test_m.get('accuracy', 0)*100:.1f}%", "Test split", "positive")
        with c2:
            render_stat_box("Precision", f"{test_m.get('precision', 0)*100:.1f}%", "Class 1 (Degradation)", "positive")
        with c3:
            render_stat_box("Recall", f"{test_m.get('recall', 0)*100:.1f}%", "Defect catch rate", "positive")
        with c4:
            render_stat_box("F1-Score", f"{test_m.get('f1', 0):.3f}", "Harmonic mean", "positive")
        with c5:
            render_stat_box("ROC-AUC", f"{test_m.get('roc_auc', 0):.3f}", "Discrimination power", "positive")

        st.markdown("##### Candidate Model Comparison (5-Fold Stratified Validation)")
        val_data = report.get("candidate_val_metrics", {})
        if val_data:
            val_df = pd.DataFrame(val_data).T.reset_index().rename(columns={"index": "Candidate Algorithm"})
            st.dataframe(val_df, width="stretch", hide_index=True)


# ==========================================================================
# TAB 5: Metric Specifications & Developer Guide
# ==========================================================================
with tab_docs:
    st.markdown("#### Metric Specifications & Interpretation Guide")
    st.markdown("""
    This system applies static code analysis and machine learning to evaluate whether a code change introduces maintainability debt.

    ##### 1. Metric Definitions
    | Metric | Formula / Source | Normal Range | Interpretation |
    | :--- | :--- | :--- | :--- |
    | **Maintainability Index (MI)** | Radon / SEI Formula: $171 - 5.2\\ln(V) - 0.23CC - 16.2\\ln(LOC) + 50\\sin(\\sqrt{2.4 \\cdot r_{comment}})$ | 85 – 100 | Higher is better. Rewards clean modular code and penalizes large unformatted blocks. |
    | **Cyclomatic Complexity (CC)** | McCabe CC: $E - N + 2P$ (Decision points + 1) | 1 – 10 | Measures independent execution paths. Scores above 15 require refactoring into smaller helper functions. |
    | **Max Nesting Depth** | AST block traversal | 1 – 3 | Deep nesting ($>4$) indicates arrowhead anti-patterns. Refactor with early return guard clauses. |
    | **Halstead Volume ($V$)** | $N \\log_2(\\eta)$ (Tokens & Operators) | Scale relative | Measures vocabulary and cognitive size of the code. |
    | **Code Churn** | Added + Deleted lines | Proportional to PR scope | High churn coupled with low test additions represents elevated defect risk. |

    ##### 2. Interpreting Predictions & Probabilities
    - **Safe / Likely Improvement (Class 0):** The model is confident that the commit does not increase cyclomatic complexity or degrade the maintainability index.
    - **Maintainability Risk (Class 1):** Indicates that the changes add significant branching, deep nesting, or unmanaged complexity.
    - **SHAP Feature Attribution:** Explains each individual prediction by quantifying how much each structural feature raised or lowered the risk score from the baseline.
    """)
