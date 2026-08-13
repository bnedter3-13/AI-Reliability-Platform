"""
app/api/routes.py

All API endpoints, kept in one router and included from main.py.
"""

import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import EvaluationRecord
from app.schemas.health_check import HealthCheckRequest, HealthCheckResponse, EvaluationResponse, RagEvaluationResponse
from app.evaluation.evaluator import evaluate_answer, EVALUATOR_VERSION
from app.evaluation.rag_evaluator import run_full_rag_evaluation
from app.root_cause.analyzer import analyze_root_cause
from app.root_cause.smart_recommendation import generate_smart_recommendation
from app.monitoring.metrics import get_metrics_summary
from app.monitoring.drift_detector import check_drift
from app.agents.analysis_agent import analyze_recent_patterns
from app.comparison.model_comparator import compare_models, AVAILABLE_MODELS
from app.schemas.model_comparison import ModelComparisonRequest, ModelComparisonResponse, ModelComparisonResultResponse
from app.evaluation.prompt_evaluator import evaluate_prompt
from app.schemas.prompt_evaluation import PromptEvaluationRequest, PromptEvaluationResponse
from app.mlops.version_tracker import list_versions, compare_versions, generate_periodic_report
from app.knowledge_base.indexing_service import index_pdf
from app.knowledge_base.verification_agent import verify_question
from app.schemas.knowledge_base import (
    KnowledgeBaseUploadResponse,
    KnowledgeBaseVerifyRequest,
    KnowledgeBaseVerifyResponse,
)

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

    # RAG Evaluation (Component 9) — Context Relevance always; Context Precision/Recall
    # too if a reference_answer was provided. Only run when there are contexts to score,
    # and only when the answer wasn't a clean pass (keeps API latency/cost down).
    rag_report = None
    context_relevance_avg = None
    if context_texts and result.status != "pass":
        try:
            rag_report = run_full_rag_evaluation(
                question=payload.question,
                contexts=context_texts,
                reference_answer=payload.reference_answer,
            )
            context_relevance_avg = rag_report.context_relevance_avg
        except Exception as exc:
            logger.warning("RAG evaluation skipped due to error: %s", exc)

    root_cause_result = analyze_root_cause(result, context_texts, context_relevance_avg)

    # Recommendation Engine upgrade (Component 4): try a Claude-generated,
    # case-specific recommendation; silently falls back to the static one on any error.
    final_recommendation = root_cause_result.recommendation
    if root_cause_result.cause not in ("none", "unknown", "evaluator_error"):
        try:
            final_recommendation = generate_smart_recommendation(
                question=payload.question,
                answer=payload.answer,
                cause=root_cause_result.cause,
                explanation=root_cause_result.explanation,
                fallback_recommendation=root_cause_result.recommendation,
            )
        except Exception as exc:
            logger.warning("Smart recommendation call failed, using static fallback: %s", exc)

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
        recommendation=final_recommendation,
        evaluator_version=EVALUATOR_VERSION,
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
        recommendation=final_recommendation,
        rag_evaluation=RagEvaluationResponse(**rag_report.to_dict()) if rag_report else None,
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


@router.get("/analysis-report")
def get_analysis_report(project_id: Optional[str] = None, limit: int = 20, db: Session = Depends(get_db)):
    """
    AI Analysis Agent (Component 3): looks across the most recent `limit` evaluations
    as a batch and reports on recurring patterns (e.g. a root cause that keeps repeating,
    a topic that consistently fails) — insight that no single evaluation's explanation
    field can surface on its own.
    """
    try:
        report = analyze_recent_patterns(db, project_id=project_id, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return report.to_dict()


@router.get("/models")
def list_available_models():
    """List the models available for comparison (Component 8)."""
    return AVAILABLE_MODELS


@router.post("/model-comparison", response_model=ModelComparisonResponse)
def create_model_comparison(payload: ModelComparisonRequest):
    """
    Model Comparison (Component 8): runs the same question through multiple models,
    evaluates each answer with the standard judge, and returns quality/speed/cost
    for each — so a team can pick the best-fit model for their use case.
    """
    context_texts = [c.text for c in payload.contexts]

    try:
        results = compare_models(
            question=payload.question,
            contexts=context_texts,
            reference_answer=payload.reference_answer,
            model_ids=payload.model_ids,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ModelComparisonResponse(
        question=payload.question,
        results=[ModelComparisonResultResponse(**r.to_dict()) for r in results],
    )


@router.post("/prompt-evaluation", response_model=PromptEvaluationResponse)
def create_prompt_evaluation(payload: PromptEvaluationRequest):
    """
    Prompt Evaluation (Component 7): standalone developer tool — evaluates the
    quality of a PROMPT itself (clarity, completeness, hallucination risk), not
    an answer. Run this while writing/revising a prompt, before it ships.
    """
    try:
        result = evaluate_prompt(payload.prompt_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PromptEvaluationResponse(**result.to_dict())


@router.get("/mlops/versions")
def get_versions(project_id: Optional[str] = None, db: Session = Depends(get_db)):
    """
    MLOps Integration (Component 10): list every evaluator version seen so far
    (judge prompt + model, tagged automatically on each evaluation) with sample counts.
    """
    return list_versions(db, project_id=project_id)


@router.get("/mlops/compare")
def get_version_comparison(
    version_a: str, version_b: str, project_id: Optional[str] = None, db: Session = Depends(get_db)
):
    """
    MLOps Integration (Component 10): regression-test two evaluator versions against
    each other — compares aggregate faithfulness/pass-rate to catch a prompt or model
    change that quietly made things worse.
    """
    return compare_versions(db, version_a=version_a, version_b=version_b, project_id=project_id).to_dict()


@router.get("/mlops/report")
def get_periodic_report(project_id: Optional[str] = None, db: Session = Depends(get_db)):
    """
    MLOps Integration (Component 10): a combined snapshot meant to be pulled on a
    schedule — current version's performance, drift status, and comparison against
    the previous version, with a suggested next action.
    """
    return generate_periodic_report(db, project_id=project_id)


@router.post("/knowledge-base/upload", response_model=KnowledgeBaseUploadResponse)
async def upload_knowledge_base_document(project_id: str = Form(...), file: UploadFile = File(...)):
    """
    Knowledge Base + Retrieval Verification Agent (Component 11): ingest a PDF into
    a project's knowledge base — extracts text, chunks it, embeds the chunks, and
    stores them in the local Chroma vector store.
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = index_pdf(project_id=project_id, file_path=tmp_path, original_filename=file.filename)
    finally:
        os.remove(tmp_path)

    return KnowledgeBaseUploadResponse(
        project_id=result.project_id,
        doc_id=result.doc_id,
        filename=result.filename,
        num_chunks=result.num_chunks,
    )


@router.post("/knowledge-base/verify", response_model=KnowledgeBaseVerifyResponse)
def verify_knowledge_base_question(payload: KnowledgeBaseVerifyRequest):
    """
    Knowledge Base + Retrieval Verification Agent (Component 11): independently
    checks whether `question` is actually supported by the project's indexed
    documents — separate from whatever a RAG app under test claims it retrieved.
    """
    result = verify_question(
        project_id=payload.project_id, question=payload.question, n_results=payload.n_results
    )
    return KnowledgeBaseVerifyResponse(
        question=result.question,
        supported=result.supported,
        best_relevance_score=result.best_relevance_score,
        threshold=result.threshold,
        matched_chunks=result.matched_chunks,
    )
