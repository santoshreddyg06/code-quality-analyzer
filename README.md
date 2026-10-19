# AI-Based Code Change Quality Analyzer

An explainable machine-learning system that predicts whether a software code change
is likely to **improve** or **negatively affect** maintainability.

> **Important:** Predictions are *estimates*, not guarantees. This tool surfaces
> signals from historical patterns; it does not replace code review.

---

## What it does

Given a GitHub repository and a commit (or pull request), the system will:

1. Retrieve the before and after versions of changed files
2. Calculate code-quality metrics (cyclomatic complexity, lines of code, churn, etc.)
3. Compute metric differences as features
4. Predict the likely maintenance impact using an ML model
5. Explain the prediction using SHAP (SHapley Additive exPlanations)
6. Display everything in an interactive Streamlit dashboard

---

## Project Structure

`
code_quality_analyzer/
+-- src/
¦   +-- data_collection/   # GitHub API, commit/PR fetching
¦   +-- metrics/           # Radon, AST-based metric extraction
¦   +-- labeling/          # Historical label generation logic
¦   +-- features/          # Feature engineering
¦   +-- models/            # ML training and evaluation
¦   +-- explainability/    # SHAP integration
¦   +-- pipeline/          # End-to-end analysis pipeline
¦   +-- api/               # FastAPI backend
¦   +-- dashboard/         # Streamlit frontend
+-- data/
¦   +-- raw/               # Raw collected data (gitignored)
¦   +-- processed/         # Cleaned, feature-engineered dataset
¦   +-- models/            # Saved ML models (gitignored)
+-- tests/                 # Automated tests
+-- notebooks/             # Exploration and EDA notebooks
+-- docs/                  # Documentation
+-- scripts/               # Utility scripts
+-- configs/               # Configuration files
+-- .env.example           # Environment variable template
+-- requirements.txt       # Python dependencies
`

---

## Quick Start

### Prerequisites
- Python 3.10+
- A GitHub Personal Access Token (PAT) with public_repo read access

### Setup

`ash
# 1. Clone the repository
git clone <your-repo-url>
cd code_quality_analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GITHUB_TOKEN
`

### Run the dashboard

`ash
# Start the FastAPI backend
uvicorn src.api.main:app --reload --port 8000

# In a second terminal, start the Streamlit dashboard
streamlit run src/dashboard/app.py
`

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| GitHub API | PyGithub |
| Code Metrics | Radon, Python AST |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn, XGBoost |
| Explainability | SHAP |
| API Backend | FastAPI |
| Dashboard | Streamlit, Plotly |
| Testing | pytest |

---

## Development Phases

- [x] Phase 1: Project Setup
- [ ] Phase 2: Historical Data Collection
- [ ] Phase 3: Metric Extraction
- [ ] Phase 4: Label Generation
- [ ] Phase 5: Dataset Preparation
- [ ] Phase 6: Machine Learning
- [ ] Phase 7: SHAP Explainability
- [ ] Phase 8: Analysis Pipeline
- [ ] Phase 9: FastAPI Backend
- [ ] Phase 10: Streamlit Dashboard
- [ ] Phase 11: Deployment
- [ ] Phase 12: Final Testing

---

## Limitations

- Predictions are statistical estimates derived from historical patterns
- Labels are based on heuristics applied to real GitHub history (documented in docs/labeling_strategy.md)
- The model works best on Python repositories
- Test coverage data is only included when reliably available in CI history

---

## License

MIT
