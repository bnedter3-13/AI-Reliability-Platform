"""
app/schemas/health_check.py

Pydantic models describing the shape of requests/responses for the
/api/v1/health-checks endpoint (a "health check" = one evaluation of a RAG answer).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ContextItem(BaseModel):
    text: str = Field(..., description="The raw text of a single retrieved context chunk.")
    source: Optional[str] = Field(None, description="Optional source identifier (doc id, URL, etc.)")


class HealthCheckRequest(BaseModel):
    project_id: str = Field(..., description="Identifier of the AI system/project being evaluated.")
    question: str = Field(..., description="The user question that was asked.")
    answer: str = Field(..., description="The answer produced by the AI system under test.")
    contexts: List[ContextItem] = Field(default_factory=list, description="Contexts retrieved by RAG, if any.")
    reference_answer: Optional[str] = Field(None, description="Optional known-correct answer, if available.")
    model_name: Optional[str] = Field(None, description="Which model produced the answer (for Model Comparison).")


class EvaluationResponse(BaseModel):
    correctness_score: float
    faithfulness_score: float
    hallucination_risk: float
    status: str
    explanation: str
    latency_ms: float


class HealthCheckResponse(BaseModel):
    project_id: str
    question: str
    answer: str
    evaluation: EvaluationResponse
    root_cause: Optional[str] = None
    recommendation: Optional[str] = None
