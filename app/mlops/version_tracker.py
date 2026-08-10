"""
app/mlops/version_tracker.py

MLOps Integration (Component 10): tracks which evaluator version (judge prompt +
model, see EVALUATOR_VERSION in evaluator.py) produced each stored evaluation, and
lets the team compare aggregate performance across versions — e.g. "did switching
from claude-sonnet-5 to a new prompt version improve or hurt faithfulness scores?"

This intentionally reuses the existing evaluations table (no new database needed)
by grouping on the evaluator_version column added to EvaluationRecord.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.models import EvaluationRecord
from app.monitoring.drift_detector import check_drift


@dataclass
class VersionSummary:
    evaluator_version: str
    sample_size: int
    avg_correctness: float
    avg_faithfulness: float
    avg_hallucination_risk: float
    avg_latency_ms: float
    pass_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VersionComparison:
    version_a: VersionSummary
    version_b: VersionSummary
    faithfulness_delta: float  # version_b - version_a; positive = improvement
    pass_rate_delta: float
    verdict: str  # short human-readable takeaway

    def to_dict(self) -> dict:
        return {
            "version_a": self.version_a.to_dict(),
            "version_b": self.version_b.to_dict(),
            "faithfulness_delta": self.faithfulness_delta,
            "pass_rate_delta": self.pass_rate_delta,
            "verdict": self.verdict,
        }


def list_versions(db: Session, project_id: Optional[str] = None) -> List[dict]:
    """List every evaluator_version seen in the evaluations table, with sample counts,
    most recent version first. Rows saved before this feature existed have a NULL
    version and are grouped under 'unknown'."""
    query = db.query(
        EvaluationRecord.evaluator_version,
        func.count(EvaluationRecord.id),
        func.max(EvaluationRecord.created_at),
    )
    if project_id:
        query = query.filter(EvaluationRecord.project_id == project_id)
    query = query.group_by(EvaluationRecord.evaluator_version).order_by(func.max(EvaluationRecord.created_at).desc())

    return [
        {
            "evaluator_version": version or "unknown (pre-versioning)",
            "sample_size": count,
            "last_used": last_used.isoformat() if last_used else None,
        }
        for version, count, last_used in query.all()
    ]


def _summarize_version(db: Session, evaluator_version: str, project_id: Optional[str] = None) -> VersionSummary:
    query = db.query(EvaluationRecord)
    if evaluator_version == "unknown (pre-versioning)":
        query = query.filter(EvaluationRecord.evaluator_version.is_(None))
    else:
        query = query.filter(EvaluationRecord.evaluator_version == evaluator_version)
    if project_id:
        query = query.filter(EvaluationRecord.project_id == project_id)

    total = query.count()
    if total == 0:
        return VersionSummary(evaluator_version, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    avg_correctness = query.with_entities(func.avg(EvaluationRecord.correctness_score)).scalar() or 0.0
    avg_faithfulness = query.with_entities(func.avg(EvaluationRecord.faithfulness_score)).scalar() or 0.0
    avg_hallucination = query.with_entities(func.avg(EvaluationRecord.hallucination_risk)).scalar() or 0.0
    avg_latency = query.with_entities(func.avg(EvaluationRecord.latency_ms)).scalar() or 0.0
    pass_count = query.filter(EvaluationRecord.status == "pass").count()

    return VersionSummary(
        evaluator_version=evaluator_version,
        sample_size=total,
        avg_correctness=round(avg_correctness, 3),
        avg_faithfulness=round(avg_faithfulness, 3),
        avg_hallucination_risk=round(avg_hallucination, 3),
        avg_latency_ms=round(avg_latency, 1),
        pass_rate=round(pass_count / total, 3),
    )


def compare_versions(
    db: Session, version_a: str, version_b: str, project_id: Optional[str] = None
) -> VersionComparison:
    """Compare aggregate performance between two evaluator versions (regression testing)."""
    summary_a = _summarize_version(db, version_a, project_id)
    summary_b = _summarize_version(db, version_b, project_id)

    faithfulness_delta = round(summary_b.avg_faithfulness - summary_a.avg_faithfulness, 3)
    pass_rate_delta = round(summary_b.pass_rate - summary_a.pass_rate, 3)

    if summary_a.sample_size == 0 or summary_b.sample_size == 0:
        verdict = "Not enough data for one or both versions to draw a conclusion yet."
    elif faithfulness_delta > 0.05 and pass_rate_delta >= 0:
        verdict = f"Version B looks like an improvement (+{faithfulness_delta} faithfulness)."
    elif faithfulness_delta < -0.05 or pass_rate_delta < -0.05:
        verdict = f"Version B looks WORSE ({faithfulness_delta} faithfulness, {pass_rate_delta} pass rate) — consider reverting."
    else:
        verdict = "No significant difference detected between the two versions."

    return VersionComparison(
        version_a=summary_a,
        version_b=summary_b,
        faithfulness_delta=faithfulness_delta,
        pass_rate_delta=pass_rate_delta,
        verdict=verdict,
    )


def generate_periodic_report(db: Session, project_id: Optional[str] = None) -> dict:
    """
    A combined MLOps snapshot: current version's aggregate performance, drift status,
    and — if more than one version has been used — a comparison against the
    second-most-recently-used version. Meant to be pulled on a schedule (e.g. daily)
    by whatever cron/orchestration the team wires up outside this app.
    """
    versions = list_versions(db, project_id=project_id)
    drift_report = check_drift(db, project_id=project_id)

    if not versions:
        return {
            "current_version": None,
            "current_version_summary": None,
            "drift": drift_report.__dict__,
            "version_comparison": None,
            "suggested_action": "No evaluations recorded yet.",
        }

    current_version = versions[0]["evaluator_version"]
    current_summary = _summarize_version(db, current_version, project_id)

    comparison = None
    if len(versions) > 1:
        previous_version = versions[1]["evaluator_version"]
        comparison = compare_versions(db, previous_version, current_version, project_id).to_dict()

    if drift_report.drift_detected:
        suggested_action = "Drift detected — investigate recent data/prompt changes; consider retraining or reverting."
    elif comparison and "WORSE" in comparison["verdict"]:
        suggested_action = "Latest version underperforms the previous one — consider reverting the evaluator/prompt change."
    else:
        suggested_action = "No action needed — performance is stable."

    return {
        "current_version": current_version,
        "current_version_summary": current_summary.to_dict(),
        "drift": drift_report.__dict__,
        "version_comparison": comparison,
        "suggested_action": suggested_action,
    }
