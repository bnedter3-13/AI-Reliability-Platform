"""
app/api/routes.py

All API endpoints, kept in one router and included from main.py.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import EvaluationRecord
from app.schemas.health_check import HealthCheckRequest, HealthCheckResponse, EvaluationResponse
from app.evaluation.evaluator import evaluate_answer
from app.evaluation.rag_evaluator import evaluate_context_relevance, average_context_relevance
from app.root_cause.analyzer import analyze_root_cause
from app.monitoring.metrics import get_metrics_summary
from app.monitoring.drift_detector import check_drift

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


@router.post("/health-checks", response_model=HealthCheckResponse)
def create_health_check(payload: HealthCheckRequest, db: Session = Depends(get_db)):
    """
    The core endpoint: evaluate one AI answer, determine a root cause if it failed,
    persist the result, and return everything to the caller.
    """
    context_texts = [c.text for c in payload.contexts]

    try:
        result = evaluate_answer(
            answer=payload.answer,
            contexts=context_texts,
            reference_answer=payload.reference_answer,
            question=payload.question,
        )
    except RuntimeError as exc:
        # Most likely cause: ANTHROPIC_API_KEY is not configured. Surface this clearly
        # instead of letting it bubble up as an opaque 500.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Basic RAG evaluation (Component 9) — only run if there are contexts to score,
    # and only when the answer wasn't a clean pass (keeps API latency/cost down).
    context_relevance_avg = None
    if context_texts and result.status != "pass":
        try:
            relevance_results = evaluate_context_relevance(payload.question, context_texts)
            context_relevance_avg = average_context_relevance(relevance_results)
        except Exception as exc:
            logger.warning("Context relevance scoring skipped due to error: %s", exc)

    root_cause_result = analyze_root_cause(result, context_texts, context_relevance_avg)

    record = EvaluationRecord(
        project_id=payload.project_id,
        question=payload.question,
        answer=payload.answer,
        model_name=payload.model_name,
        correctness_score=result.correctness_score,
        faithfulness_score=result.faithfulness_score,
        hallucination_risk=result.hallucination_risk,
        status=result.status,
        explanation=result.explanation,
        latency_ms=result.latency_ms,
        root_cause=root_cause_result.cause,
        recommendation=root_cause_result.recommendation,
    )
    db.add(record)
    db.commit()

    return HealthCheckResponse(
        project_id=payload.project_id,
        question=payload.question,
        answer=payload.answer,
        evaluation=EvaluationResponse(
            correctness_score=result.correctness_score,
            faithfulness_score=result.faithfulness_score,
            hallucination_risk=result.hallucination_risk,
            status=result.status,
            explanation=result.explanation,
            latency_ms=result.latency_ms,
        ),
        root_cause=root_cause_result.cause,
        recommendation=root_cause_result.recommendation,
    )


@router.get("/metrics")
def get_metrics(project_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Monitoring (Component 5): aggregate stats, optionally scoped to one project."""
    return get_metrics_summary(db, project_id=project_id)


@router.get("/drift")
def get_drift(project_id: Optional[str] = None, window_days: int = 7, db: Session = Depends(get_db)):
    """Drift Detection (Component 6): recent vs. previous performance window."""
    return check_drift(db, project_id=project_id, window_days=window_days)


@router.get("/evaluations")
def list_evaluations(project_id: Optional[str] = None, limit: int = 50, db: Session = Depends(get_db)):
    """List recent evaluations — useful for the Dashboard's detail view."""
    query = db.query(EvaluationRecord).order_by(EvaluationRecord.created_at.desc())
    if project_id:
        query = query.filter(EvaluationRecord.project_id == project_id)
    records = query.limit(limit).all()
    return [
        {
            "id": r.id,
            "project_id": r.project_id,
            "question": r.question,
            "status": r.status,
            "faithfulness_score": r.faithfulness_score,
            "hallucination_risk": r.hallucination_risk,
            "root_cause": r.root_cause,
            "recommendation": r.recommendation,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]
