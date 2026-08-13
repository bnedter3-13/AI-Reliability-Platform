# AI Reliability Platform (AI Doctor)

A platform that continuously monitors, evaluates, and improves the reliability of AI/RAG
systems — acting as a "Quality Engineer" for AI. Built as the Capstone Project for the
Samsung Innovation Campus (SIC) AI program by team **Nexus AI**.

As AI adoption grows, the hard part isn't building the model — it's knowing whether it:
- hallucinates,
- relies on the right sources,
- stays reliable over time, and
- can be diagnosed quickly when something goes wrong.

This platform wraps an LLM-as-a-Judge evaluation pipeline (using Claude) around any RAG or
AI system: send it a question/answer/context triple and it scores correctness and
faithfulness, classifies *why* a bad answer failed, suggests a fix, and rolls all of that up
into dashboards for monitoring, drift detection, and cross-model/cross-version comparison.

All 11 components below are fully implemented — not stubs — and are wired together through
one FastAPI app (`app/main.py`) and one router (`app/api/routes.py`).

---

## Table of Contents

- [Components](#components)
  1. [Core Evaluation (LLM-as-a-Judge)](#1-core-evaluation-llm-as-a-judge)
  2. [Root Cause Analysis](#2-root-cause-analysis)
  3. [AI Analysis Agent](#3-ai-analysis-agent)
  4. [Recommendation Engine](#4-recommendation-engine)
  5. [Monitoring](#5-monitoring)
  6. [Drift Detection](#6-drift-detection)
  7. [Prompt Evaluation](#7-prompt-evaluation)
  8. [Model Comparison](#8-model-comparison)
  9. [RAG Evaluation](#9-rag-evaluation)
  10. [MLOps Integration](#10-mlops-integration)
  11. [Knowledge Base + Retrieval Verification Agent](#11-knowledge-base--retrieval-verification-agent)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Local Quick Start (SQLite)](#local-quick-start-sqlite)
  - [Docker (API + PostgreSQL)](#docker-api--postgresql)
  - [PostgreSQL as the Production Database](#postgresql-as-the-production-database)
  - [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Dashboard](#dashboard)

---

## Components

### 1. Core Evaluation (LLM-as-a-Judge)

**What it does:** The core pipeline. Scores a single AI-generated answer for
`correctness_score`, `faithfulness_score`, and `hallucination_risk` (all 0.0–1.0) using
Claude as an impartial judge, and assigns an overall `status` of `pass`, `warning`, or
`fail`. Implemented in [`app/evaluation/evaluator.py`](app/evaluation/evaluator.py) as
`evaluate_answer()`, and exposed as the platform's main endpoint. Every call to this
endpoint also runs Root Cause Analysis (#2), the Recommendation Engine (#4), and — for
non-passing answers with contexts — RAG Evaluation (#9), then persists everything to the
database in one shot.

**Usage — API:**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/health-checks \
  -H "Content-Type: application/json" \
  -d '{
        "project_id": "support-bot",
        "question": "What is our refund policy?",
        "answer": "We offer refunds within 30 days of purchase, no questions asked.",
        "contexts": [
          {"text": "Refunds are available within 30 days of purchase.", "source": "policy.md"}
        ],
        "reference_answer": "Refunds within 30 days of purchase.",
        "model_name": "claude-sonnet-5"
      }'
```

**Usage — Python:**

```python
from app.evaluation.evaluator import evaluate_answer

result = evaluate_answer(
    answer="We offer refunds within 30 days of purchase.",
    contexts=["Refunds are available within 30 days of purchase."],
    reference_answer="Refunds within 30 days of purchase.",
    question="What is our refund policy?",
)
print(result.status, result.faithfulness_score)
```

---

### 2. Root Cause Analysis

**What it does:** For any non-passing evaluation, classifies *why* it failed —
`no_context`, `poor_retrieval`, `hallucination`, `unfaithful_answer`, `low_correctness`, or
`unknown` — using cheap, explainable rules over the scores already computed (no extra LLM
call). Implemented in
[`app/root_cause/analyzer.py`](app/root_cause/analyzer.py) as `analyze_root_cause()`. It
isn't a separate endpoint — it runs automatically inside `POST /api/v1/health-checks` and
its result appears in the `root_cause` field of that response.

**Usage — Python:**

```python
from app.root_cause.analyzer import analyze_root_cause

root_cause = analyze_root_cause(result, contexts=["Refunds are available within 30 days."], context_relevance_avg=0.3)
print(root_cause.cause, root_cause.recommendation)
```

---

### 3. AI Analysis Agent

**What it does:** Looks *across* the most recent evaluations as a batch (not one at a
time) and asks Claude to find recurring patterns — e.g. "hallucination keeps happening on
questions about pricing" — the kind of insight a single evaluation's explanation can't
surface. Implemented in
[`app/agents/analysis_agent.py`](app/agents/analysis_agent.py) as
`analyze_recent_patterns()`.

**Usage — API:**

```bash
curl "http://127.0.0.1:8000/api/v1/analysis-report?project_id=support-bot&limit=20"
```

**Usage — Python:**

```python
from app.agents.analysis_agent import analyze_recent_patterns
from app.database.connection import SessionLocal

db = SessionLocal()
report = analyze_recent_patterns(db, project_id="support-bot", limit=20)
print(report.dominant_issue, report.pattern_detected, report.severity)
```

---

### 4. Recommendation Engine

**What it does:** Turns a classified root cause into an actionable fix. Starts from a
free, always-on static lookup table (`ROOT_CAUSES`/`RECOMMENDATIONS` in
[`app/root_cause/analyzer.py`](app/root_cause/analyzer.py)), then — for failures with a
known cause — upgrades it with a Claude-generated, case-specific suggestion (referencing
the actual question/answer) via
[`app/root_cause/smart_recommendation.py`](app/root_cause/smart_recommendation.py)'s
`generate_smart_recommendation()`. It silently falls back to the static recommendation on
any error, so it never breaks a request. Like Root Cause Analysis, this has no dedicated
endpoint — it populates the `recommendation` field of `POST /api/v1/health-checks`.

**Usage — Python:**

```python
from app.root_cause.smart_recommendation import generate_smart_recommendation

recommendation = generate_smart_recommendation(
    question="What is our refund policy?",
    answer="We offer refunds within 30 days of purchase, no questions asked.",
    cause="unfaithful_answer",
    explanation="The answer partially diverges from what the contexts state.",
    fallback_recommendation="Review the prompt for ambiguous instructions.",
)
```

---

### 5. Monitoring

**What it does:** Aggregate statistics over all stored evaluations — total requests,
average correctness/faithfulness/hallucination-risk/latency, and pass rate — optionally
scoped to one `project_id`. Implemented in
[`app/monitoring/metrics.py`](app/monitoring/metrics.py) as `get_metrics_summary()`.
Powers the "Score Trends" section of the dashboard.

**Usage — API:**

```bash
curl "http://127.0.0.1:8000/api/v1/metrics?project_id=support-bot"
```

---

### 6. Drift Detection

**What it does:** Compares average faithfulness in a recent time window (default 7 days)
against the window before it, and flags `drift_detected: true` when the drop exceeds
`DRIFT_ALERT_THRESHOLD`. Implemented in
[`app/monitoring/drift_detector.py`](app/monitoring/drift_detector.py) as `check_drift()`.

**Usage — API:**

```bash
curl "http://127.0.0.1:8000/api/v1/drift?project_id=support-bot&window_days=7"
```

---

### 7. Prompt Evaluation

**What it does:** A standalone developer tool that critiques the quality of a *prompt*
itself (not an answer) — clarity, completeness, and hallucination risk — before it ever
ships to production, plus a concrete `suggested_rewrite`. Implemented in
[`app/evaluation/prompt_evaluator.py`](app/evaluation/prompt_evaluator.py) as
`evaluate_prompt()`.

**Usage — API:**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/prompt-evaluation \
  -H "Content-Type: application/json" \
  -d '{"prompt_text": "You are a helpful assistant. Answer the user question."}'
```

**Usage — Python:**

```python
from app.evaluation.prompt_evaluator import evaluate_prompt

result = evaluate_prompt("You are a helpful assistant. Answer the user question.")
print(result.clarity_score, result.issues, result.suggested_rewrite)
```

---

### 8. Model Comparison

**What it does:** Runs the same question through multiple models — across Anthropic,
OpenAI, Google Gemini, and Qwen (via OpenRouter) — evaluates every generated answer with
the same Claude judge so scores stay comparable, and reports quality, speed, and estimated
cost per model. Implemented in
[`app/comparison/model_comparator.py`](app/comparison/model_comparator.py) as
`compare_models()`; the full model catalog lives in `AVAILABLE_MODELS` in that file.

**Usage — API:**

```bash
# List available models
curl http://127.0.0.1:8000/api/v1/models

# Compare a subset of them on one question
curl -X POST http://127.0.0.1:8000/api/v1/model-comparison \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What is our refund policy?",
        "contexts": [{"text": "Refunds are available within 30 days of purchase."}],
        "reference_answer": "Refunds within 30 days of purchase.",
        "model_ids": ["claude-sonnet-5", "gpt-4o-mini", "gemini-3.6-flash"]
      }'
```

Omit `model_ids` to run every model in `AVAILABLE_MODELS` that you have an API key for. A
model that fails (missing key, provider error) shows up in the results with its `error`
field set, rather than stopping the others.

---

### 9. RAG Evaluation

**What it does:** Scores the *retrieval* side of a RAG pipeline, separately from the
answer itself:
- **Context Relevance** — is each retrieved passage relevant to the question? (always computed)
- **Context Precision** — was each retrieved passage actually useful for the reference answer? (needs a `reference_answer`)
- **Context Recall** — did retrieval, as a whole, cover everything needed? (needs a `reference_answer`)

Implemented in [`app/evaluation/rag_evaluator.py`](app/evaluation/rag_evaluator.py) as
`run_full_rag_evaluation()`. It's automatically triggered inside
`POST /api/v1/health-checks` (only when contexts were provided and the answer wasn't a
clean pass, to save latency/cost) and shows up in that response's `rag_evaluation` field —
but it can also be called directly for a standalone RAG retrieval audit:

**Usage — Python:**

```python
from app.evaluation.rag_evaluator import run_full_rag_evaluation

report = run_full_rag_evaluation(
    question="What is our refund policy?",
    contexts=["Refunds are available within 30 days of purchase.", "We ship worldwide."],
    reference_answer="Refunds within 30 days of purchase.",
)
print(report.context_relevance_avg, report.context_precision_avg, report.context_recall_score)
```

---

### 10. MLOps Integration

**What it does:** Tracks which "evaluator version" (judge prompt version + model name,
tagged automatically as `EVALUATOR_VERSION` on every evaluation) produced each stored
result, and lets the team regression-test performance across versions — e.g. "did
switching judge models improve or hurt faithfulness scores?" Implemented in
[`app/mlops/version_tracker.py`](app/mlops/version_tracker.py) with three functions:
`list_versions()`, `compare_versions()`, and `generate_periodic_report()` (a combined
snapshot of current performance + drift + version comparison, meant to be pulled on a
schedule).

**Usage — API:**

```bash
# See every evaluator version used so far, with sample counts
curl "http://127.0.0.1:8000/api/v1/mlops/versions?project_id=support-bot"

# Regression-test two versions against each other
curl "http://127.0.0.1:8000/api/v1/mlops/compare?version_a=claude-sonnet-5:1.0.0&version_b=claude-opus-4-8:1.0.0"

# Pull a combined daily/periodic snapshot
curl "http://127.0.0.1:8000/api/v1/mlops/report?project_id=support-bot"
```

---

### 11. Knowledge Base + Retrieval Verification Agent

**What it does:** Independently checks whether a question is actually supported by a
project's indexed source documents — separate from, and a check on, whatever a RAG app
under test claims it retrieved. A PDF is uploaded and run through an ingest pipeline
(extract text → chunk → embed → store) into a local Chroma vector store, one collection
per `project_id`. A question is then verified by embedding it, querying that collection for
the best-matching chunk, and comparing its relevance (`1 - cosine distance`) against
`RELEVANCE_THRESHOLD` (0.65 — chunking mixes numeric data with prose, which dilutes
embedding relevance even for chunks that clearly answer the question, so a stricter
threshold produced false negatives on real documents). Uses Chroma's bundled local
embedding model (ONNX MiniLM-L6-v2), so no external embedding API/key is required.
Implemented across
[`app/knowledge_base/document_loader.py`](app/knowledge_base/document_loader.py) (PDF text
extraction via PyMuPDF),
[`app/knowledge_base/chunker.py`](app/knowledge_base/chunker.py) (fixed-size overlapping
chunking),
[`app/knowledge_base/vector_store.py`](app/knowledge_base/vector_store.py) (`KnowledgeBaseStore`,
the Chroma wrapper),
[`app/knowledge_base/indexing_service.py`](app/knowledge_base/indexing_service.py)
(`index_pdf()`, the full ingest pipeline), and
[`app/knowledge_base/verification_agent.py`](app/knowledge_base/verification_agent.py)
(`verify_question()`).

**Usage — API:**

```bash
# Upload and index a PDF into a project's knowledge base
curl -X POST http://127.0.0.1:8000/api/v1/knowledge-base/upload \
  -F "project_id=support-bot" \
  -F "file=@policy.pdf"

# Verify whether a question is actually supported by that knowledge base
curl -X POST http://127.0.0.1:8000/api/v1/knowledge-base/verify \
  -H "Content-Type: application/json" \
  -d '{"project_id": "support-bot", "question": "What is our refund policy?"}'
```

**Usage — Python:**

```python
from app.knowledge_base.indexing_service import index_pdf
from app.knowledge_base.verification_agent import verify_question

index_pdf(project_id="support-bot", file_path="policy.pdf", original_filename="policy.pdf")

result = verify_question(project_id="support-bot", question="What is our refund policy?")
print(result.supported, result.best_relevance_score, result.matched_chunks)
```

---

### Other useful endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Liveness message |
| `GET` | `/health` | Health check for the service itself |
| `GET` | `/dashboard` | Serves the built-in HTML dashboard |
| `GET` | `/docs` | Interactive Swagger UI for every endpoint above |
| `GET` | `/api/v1/evaluations` | List recent stored evaluations (backs the dashboard's "Recent Evaluations" table) |

---

## Project Structure

```
app/
├── main.py                        # FastAPI app entrypoint, mounts the router, creates tables on startup
├── config.py                      # Loads .env / environment variables into `settings`
├── schemas/                       # Pydantic request/response models
│   ├── health_check.py
│   ├── model_comparison.py
│   └── prompt_evaluation.py
├── evaluation/
│   ├── prompts.py                 # Judge prompt templates + JUDGE_PROMPT_VERSION
│   ├── evaluator.py                # (1) Core Evaluation — evaluate_answer()
│   ├── prompt_evaluator.py         # (7) Prompt Evaluation — evaluate_prompt()
│   └── rag_evaluator.py            # (9) RAG Evaluation — run_full_rag_evaluation()
├── root_cause/
│   ├── analyzer.py                 # (2) Root Cause Analysis + static Recommendation Engine
│   └── smart_recommendation.py     # (4) Claude-generated recommendation upgrade
├── agents/
│   └── analysis_agent.py           # (3) AI Analysis Agent — analyze_recent_patterns()
├── monitoring/
│   ├── metrics.py                  # (5) Monitoring — get_metrics_summary()
│   └── drift_detector.py           # (6) Drift Detection — check_drift()
├── comparison/
│   └── model_comparator.py         # (8) Model Comparison — compare_models()
├── mlops/
│   └── version_tracker.py          # (10) MLOps Integration — versions/compare/report
├── knowledge_base/
│   ├── document_loader.py          # (11) PDF text extraction (PyMuPDF)
│   ├── chunker.py                  # (11) Fixed-size overlapping text chunking
│   ├── vector_store.py             # (11) KnowledgeBaseStore — Chroma wrapper
│   ├── indexing_service.py         # (11) index_pdf() — full ingest pipeline
│   └── verification_agent.py       # (11) verify_question() — retrieval verification
├── database/
│   ├── models.py                   # SQLAlchemy models (EvaluationRecord)
│   └── connection.py               # Engine/session setup + lightweight auto-migration
├── dashboard/
│   └── index.html                  # Built-in dashboard (served at /dashboard)
└── api/
    └── routes.py                   # All API endpoints, wired to the modules above
tests/
├── conftest.py                     # Shared in-memory SQLite fixture (`db_session`)
├── test_evaluator.py
├── test_model_comparator.py
├── test_version_tracker.py
├── test_rag_evaluator.py
├── test_prompt_evaluator.py
├── test_analysis_agent.py
├── test_chunker.py
├── test_vector_store.py
└── test_verification_agent.py
scripts/
├── index_pdf_demo.py               # (11) Manual-test script: generates + indexes a sample PDF
├── seed_scenarios.py               # Seeds data/seed_scenarios.json (evaluator v1.0.0 baseline)
├── seed_scenarios_v2.py            # Seeds data/seed_scenarios_v2.json (judge prompt v1.1.0)
├── seed_scenarios_v3.py            # Seeds data/seed_scenarios_v3.json (generation prompt hardening)
├── seed_scenarios_v4.py            # Seeds data/seed_scenarios_v4.json (retry-with-clarification)
└── seed_scenarios_v5.py            # Seeds data/seed_scenarios_v5.json (ThinkingBlock fix + 2nd retry)
data/
├── seed_scenarios*.json            # Seed data consumed by scripts/seed_scenarios*.py
└── chroma_db/                      # Local Chroma vector store (gitignored, created at runtime)
notebooks/
└── evaluator_prototype.ipynb       # Exploratory notebook
```

---

## Setup

### Local Quick Start (SQLite)

The fastest way to run the platform — no external database needed.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# then edit .env and add your ANTHROPIC_API_KEY (required — get one from console.anthropic.com)

# 4. Run the API (creates ai_reliability.db automatically on first startup)
uvicorn app.main:app --reload
```

Once running:
- Swagger UI: http://127.0.0.1:8000/docs
- Dashboard: http://127.0.0.1:8000/dashboard

### Docker (API + PostgreSQL)

`docker-compose.yml` runs the API alongside a PostgreSQL container — no local Python or
Postgres install needed.

```bash
# 1. Configure environment variables (API keys are read from .env via env_file)
cp .env.example .env
# then edit .env and add your ANTHROPIC_API_KEY

# 2. Build and start both containers
docker compose up --build
```

This starts:
- `api` — the FastAPI app on `http://localhost:8000`, with `DATABASE_URL` overridden to
  point at the `db` service (`postgresql://postgres:postgres@db:5432/ai_reliability`).
- `db` — a `postgres:16` container with a named volume (`pgdata`) so data survives restarts.

Tables (and any missing columns) are created automatically on API startup — no manual
migration step needed for a fresh database.

### PostgreSQL as the Production Database

Outside of Docker, point the app at any PostgreSQL instance by setting `DATABASE_URL` —
the `psycopg2-binary` driver is already in `requirements.txt`.

```bash
# .env
DATABASE_URL=postgresql://<user>:<password>@<host>:5432/<database>
```

On startup, `init_db()` (in
[`app/database/connection.py`](app/database/connection.py)) runs
`Base.metadata.create_all()` to create any missing tables, then a lightweight additive
migration that adds any missing *columns* (e.g. `evaluator_version`) to an existing
`evaluations` table via `ALTER TABLE` — so upgrading the app on top of real historical data
doesn't lose anything. This only handles additive changes; for renames, type changes, or
drops, use a real migration tool (e.g. Alembic) instead.

### Environment Variables

All variables are read once at startup into `settings` (see
[`app/config.py`](app/config.py)). Copy `.env.example` to `.env` and fill these in:

| Variable | Required? | Default | Used by |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | **Required** | — | The Claude judge (Components 1, 3, 4, 7, 9) and Claude as a comparison model (8). The API returns `503` on any endpoint that needs it if it's missing. |
| `OPENAI_API_KEY` | Optional | — | Model Comparison (8) — enables GPT-4o / GPT-4o-mini as comparison targets. |
| `GEMINI_API_KEY` | Optional | — | Model Comparison (8) — enables Gemini models as comparison targets. |
| `QWEN_API_KEY` | Optional | — | Model Comparison (8) — an **OpenRouter** API key (openrouter.ai), enables Qwen models via OpenRouter's OpenAI-compatible API. |
| `DATABASE_URL` | Optional | `sqlite:///./ai_reliability.db` | All persistence — set to a `postgresql://...` URL for production (see above). |
| `DRIFT_ALERT_THRESHOLD` | Optional | `0.15` | Drift Detection (6) — fraction drop in average faithfulness that triggers `drift_detected: true`. |
| `CHROMA_PERSIST_DIR` | Optional | `./data/chroma_db` | Knowledge Base + Retrieval Verification Agent (11) — where the local Chroma vector store persists on disk. |

A model with no matching API key configured simply fails for that one provider (with its
`error` field set) in Model Comparison — the rest of the request still succeeds.

> Note: the judge model itself (`MODEL_NAME`) is currently a fixed constant
> (`claude-sonnet-5`) in `app/config.py` rather than an environment variable — change it
> there if you want to run the judge on a different Claude model.

---

## Testing

The test suite uses `pytest` and deliberately avoids calling real provider APIs — no
network access or API cost required to run it. LLM-calling functions are tested by
monkeypatching the module-level Anthropic client with a fake object that returns canned
JSON, and database-backed functions run against an in-memory SQLite database (see
`tests/conftest.py`'s `db_session` fixture).

```bash
source .venv/bin/activate
pip install -r requirements.txt   # includes pytest
pytest
```

Run a single file or with more detail:

```bash
pytest tests/test_evaluator.py -v
pytest -q   # quiet summary across the whole suite
```

| File | Covers |
|---|---|
| `test_evaluator.py` | Core Evaluation (1) — JSON extraction, judge prompt building |
| `test_model_comparator.py` | Model Comparison (8) — cost estimation, per-model success/failure handling |
| `test_version_tracker.py` | MLOps Integration (10) — version listing, aggregation, regression comparisons |
| `test_rag_evaluator.py` | RAG Evaluation (9) — relevance/precision/recall scoring and error handling |
| `test_prompt_evaluator.py` | Prompt Evaluation (7) — JSON repair, score/issue coercion |
| `test_analysis_agent.py` | AI Analysis Agent (3) — batch formatting, pattern report generation |
| `test_chunker.py` | Knowledge Base (11) — chunk sizing/overlap edge cases |
| `test_vector_store.py` | Knowledge Base (11) — `KnowledgeBaseStore` add/query behavior |
| `test_verification_agent.py` | Knowledge Base (11) — relevance threshold, best-match selection |

No `ANTHROPIC_API_KEY` (or any other provider key) is required to run the suite.

---

## Dashboard

The built-in dashboard (`app/dashboard/index.html`, served at `/dashboard`) covers:
Score Trends, Root Cause Breakdown, Recent Evaluations, the AI Analysis Report, Model
Comparison, Prompt Evaluation, MLOps Version Tracking, and Knowledge Base (upload a PDF to
index it, then verify whether a question is actually supported by what's indexed).

<!--
Add screenshots here once available, e.g.:

![Score Trends](docs/screenshots/score-trends.png)
![Root Cause Breakdown](docs/screenshots/root-cause-breakdown.png)
![Model Comparison](docs/screenshots/model-comparison.png)
![MLOps Version Tracking](docs/screenshots/mlops-version-tracking.png)
-->

*(Screenshots coming soon — run the app and visit `/dashboard` to see it live.)*

---

## Tech Stack

Python, FastAPI, SQLAlchemy, PostgreSQL (SQLite for local dev), Docker, Anthropic Claude
API (with optional OpenAI, Google Gemini, and Qwen-via-OpenRouter support for Model
Comparison), ChromaDB (local vector store) and PyMuPDF (PDF extraction) for the Knowledge
Base + Retrieval Verification Agent.

## Team — Nexus AI

Capstone project, SIC AI Program (Misk x Imtiaz).
