# AI Reliability Platform (AI Doctor)

A platform that continuously monitors, evaluates, and improves the reliability of AI/RAG
systems — acting as a "Quality Engineer" for AI. Built as the Capstone Project for the
Samsung Innovation Campus (SIC) AI program by team **Nexus AI**.

## The Problem

As AI adoption grows, the real challenge is no longer building the model — it's ensuring it:
- Doesn't hallucinate
- Relies on correct sources
- Maintains consistent quality over time
- Can be diagnosed when something goes wrong

## MVP Scope (this repository)

This repo implements a focused MVP covering:

1. **Evaluation Pipeline** — LLM-as-a-Judge scoring of every answer (`app/evaluation/`)
2. **Root Cause Analysis** — classifies *why* an answer failed (`app/root_cause/`)
3. **Monitoring** — aggregate stats over stored evaluations (`app/monitoring/metrics.py`)
4. **Drift Detection** — compares recent vs. older performance windows (`app/monitoring/drift_detector.py`)
5. **RAG Evaluation (basic)** — context relevance scoring (`app/evaluation/rag_evaluator.py`)

Root Cause Analysis, Recommendation Engine, Model Comparison, Prompt Evaluation, and
MLOps Integration are represented as lightweight, extensible stubs — see comments in
each file for how to extend them further.

## Project Structure

```
app/
├── main.py                  # FastAPI app entrypoint
├── config.py                # Environment/config loading
├── schemas/                 # Pydantic request/response models
├── evaluation/
│   ├── prompts.py           # Judge prompt templates
│   ├── evaluator.py         # evaluate_answer() — core LLM-as-a-Judge logic
│   └── rag_evaluator.py     # Context relevance scoring
├── root_cause/
│   └── analyzer.py          # Maps evaluation signals -> likely root cause
├── monitoring/
│   ├── metrics.py           # Aggregate stats queries
│   └── drift_detector.py    # Time-window performance comparison
├── database/
│   ├── models.py            # SQLAlchemy models
│   └── connection.py        # DB session/engine setup
└── api/
    └── routes.py            # All API endpoints
tests/
└── test_evaluator.py
notebooks/
└── evaluator_prototype.ipynb  # exploratory notebook (add yours here)
```

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# then edit .env and add your ANTHROPIC_API_KEY

# 4. Run the API
uvicorn app.main:app --reload
```

Once running, open http://127.0.0.1:8000/docs for the interactive Swagger UI.

## Running with Docker

```bash
docker compose up --build
```

## Running Tests

```bash
pytest
```

## Tech Stack

Python, FastAPI, SQLAlchemy, PostgreSQL (SQLite for local dev), Docker, Anthropic Claude API.

## Team — Nexus AI

Capstone project, SIC AI Program (Misk x Imtiaz).
