"""
app/database/models.py

SQLAlchemy models. One row per evaluation — this table is what Monitoring (Component 5)
and Drift Detection (Component 6) both query against.
"""

import datetime
import uuid

from sqlalchemy import Column, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class EvaluationRecord(Base):
    __tablename__ = "evaluations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    model_name = Column(String, nullable=True)  # which model produced the answer (for Model Comparison)

    correctness_score = Column(Float, nullable=False)
    faithfulness_score = Column(Float, nullable=False)
    hallucination_risk = Column(Float, nullable=False)
    status = Column(String, nullable=False)
    explanation = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)

    root_cause = Column(String, nullable=True)
    recommendation = Column(Text, nullable=True)

    # MLOps Integration (Component 10): which evaluator version (judge prompt + model)
    # produced this evaluation, so performance can be compared across versions.
    evaluator_version = Column(String, nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
