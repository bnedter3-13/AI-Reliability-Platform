"""
app/monitoring/metrics.py

Monitoring (Component 5): aggregate statistics over stored evaluations.
Powers the /api/v1/metrics endpoint and, later, the Dashboard.
"""

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.models import EvaluationRecord


@dataclass
class MetricsSummary:
    total_requests: int
    avg_correctness: float
    avg_faithfulness: float
    avg_hallucination_risk: float
    avg_latency_ms: float
    pass_rate: float


def get_metrics_summary(db: Session, project_id: Optional[str] = None) -> MetricsSummary:
    """Compute aggregate stats, optionally scoped to a single project_id."""
    query = db.query(EvaluationRecord)
    if project_id:
        query = query.filter(EvaluationRecord.project_id == project_id)

    total = query.count()
    if total == 0:
        return MetricsSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    avg_correctness = query.with_entities(func.avg(EvaluationRecord.correctness_score)).scalar() or 0.0
    avg_faithfulness = query.with_entities(func.avg(EvaluationRecord.faithfulness_score)).scalar() or 0.0
    avg_hallucination = query.with_entities(func.avg(EvaluationRecord.hallucination_risk)).scalar() or 0.0
    avg_latency = query.with_entities(func.avg(EvaluationRecord.latency_ms)).scalar() or 0.0
    pass_count = query.filter(EvaluationRecord.status == "pass").count()

    return MetricsSummary(
        total_requests=total,
        avg_correctness=round(avg_correctness, 3),
        avg_faithfulness=round(avg_faithfulness, 3),
        avg_hallucination_risk=round(avg_hallucination, 3),
        avg_latency_ms=round(avg_latency, 1),
        pass_rate=round(pass_count / total, 3),
    )
