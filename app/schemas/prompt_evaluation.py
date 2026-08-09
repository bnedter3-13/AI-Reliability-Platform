"""
app/schemas/prompt_evaluation.py

Pydantic models for the /api/v1/prompt-evaluation endpoint.
"""

from typing import List
from pydantic import BaseModel, Field


class PromptEvaluationRequest(BaseModel):
    prompt_text: str = Field(..., description="The prompt text to evaluate (system prompt or generation template).")


class PromptEvaluationResponse(BaseModel):
    clarity_score: float
    completeness_score: float
    hallucination_risk_score: float
    issues: List[str]
    suggested_rewrite: str
