"""
app/monitoring/drift_detector.py

Drift Detection (Component 6): compares average faithfulness in a recent time window
against an earlier window, and flags a drop beyond DRIFT_ALERT_THRESHOLD.
"""

import datetime
from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.models import EvaluationRecord
from app.config import settings


@dataclass
class DriftReport:
    drift_detected: bool
    recent_avg_faithfulness: float
    previous_avg_faithfulness: float
    drop_percentage: float
    recent_window_days: int
    sample_size_recent: int
    sample_size_previous: int


def _avg_faithfulness_between(db: Session, start: datetime.datetime, end: datetime.datetime, project_id: str = None):
    query = db.query(func.avg(EvaluationRecord.faithfulness_score), func.count(EvaluationRecord.id)).filter(
        EvaluationRecord.created_at >= start,
        EvaluationRecord.created_at < end,
    )
    if project_id:
        query = query.filter(EvaluationRecord.project_id == project_id)
    avg, count = query.first()
    return (avg or 0.0), (count or 0)


def check_drift(db: Session, project_id: str = None, window_days: int = 7) -> DriftReport:
    """
    Compare the most recent `window_days` against the `window_days` before that.
    A meaningful drop in average faithfulness signals model/data drift.
    """
    now = datetime.datetime.utcnow()
    recent_start = now - datetime.timedelta(days=window_days)
    previous_start = now - datetime.timedelta(days=window_days * 2)

    recent_avg, recent_n = _avg_faithfulness_between(db, recent_start, now, project_id)
    previous_avg, previous_n = _avg_faithfulness_between(db, previous_start, recent_start, project_id)

    if previous_avg > 0:
        drop_percentage = round((previous_avg - recent_avg) / previous_avg, 3)
    else:
        drop_percentage = 0.0

    drift_detected = drop_percentage >= settings.DRIFT_ALERT_THRESHOLD and previous_n > 0 and recent_n > 0

    return DriftReport(
        drift_detected=drift_detected,
        recent_avg_faithfulness=round(recent_avg, 3),
        previous_avg_faithfulness=round(previous_avg, 3),
        drop_percentage=drop_percentage,
        recent_window_days=window_days,
        sample_size_recent=recent_n,
        sample_size_previous=previous_n,
    )
