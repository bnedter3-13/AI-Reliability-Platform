"""
app/schemas/model_comparison.py

Pydantic models for the /api/v1/model-comparison endpoint (Model Comparison,
Component 8): send one question, get back one evaluated answer per model.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.health_check import ContextItem


class ModelComparisonRequest(BaseModel):
    question: str = Field(..., description="The question to ask each model.")
    contexts: List[ContextItem] = Field(default_factory=list, description="Contexts retrieved by RAG, if any.")
    reference_answer: Optional[str] = Field(None, description="Optional known-correct answer, if available.")
    model_ids: Optional[List[str]] = Field(
        None,
        description="Which models to compare (see AVAILABLE_MODELS in model_comparator.py). "
                    "Omit to run every available model.",
    )


class ModelComparisonResultResponse(BaseModel):
    model_id: str
    model_label: str
    provider: str
    answer: str
    generation_latency_ms: float
    correctness_score: Optional[float] = None
    faithfulness_score: Optional[float] = None
    hallucination_risk: Optional[float] = None
    status: Optional[str] = None
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    error: Optional[str] = None


class ModelComparisonResponse(BaseModel):
    question: str
    results: List[ModelComparisonResultResponse]
