"""
scripts/seed_scenarios_v2.py

Same insert-only pattern as scripts/seed_scenarios.py, loading
data/seed_scenarios_v2.json instead. This batch reflects the real behavior
change from the GENERATION_SYSTEM_PROMPT (app/comparison/model_comparator.py)
and JUDGE_SYSTEM_PROMPT v1.1.0 (app/evaluation/prompts.py) improvements: most
models now correctly decline when context is missing/insufficient instead of
guessing, and the judge now scores those correct refusals as faithful passes.

`evaluator_version` is pulled live from EVALUATOR_VERSION in
app.evaluation.evaluator (now "claude-sonnet-5:1.1.0" following the
JUDGE_PROMPT_VERSION bump), so this batch is distinguishable from the
"claude-sonnet-5:1.0.0" rows inserted by scripts/seed_scenarios.py via
GET /api/v1/mlops/compare.

Timestamps: the real system clock can lag behind the latest timestamp already
in the database (batch 1's random jitter pushed its last row's created_at
past "now" at the time it ran), so "extending to now" isn't always literally
possible. This script anchors ANCHOR_OFFSET_HOURS after the current max
created_at in the evaluations table, and spreads forward either to the real
current time (if that's later than the anchor) or across a fixed fallback
window (if not) - see _compute_time_range().

Usage:
    python scripts/seed_scenarios_v2.py
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func

from app.database.connection import SessionLocal, init_db
from app.database.models import EvaluationRecord
from app.evaluation.evaluator import EVALUATOR_VERSION
from app.root_cause.analyzer import RECOMMENDATIONS

SCENARIOS_PATH = Path(__file__).resolve().parent.parent / "data" / "seed_scenarios_v2.json"
ANCHOR_OFFSET_HOURS = 3    # start this many hours after the existing data's latest timestamp
FALLBACK_SPAN_HOURS = 30   # spread window used if the real clock hasn't caught up to the anchor yet


def _recommendation_for(root_cause: str) -> str:
    if root_cause == "none":
        return "No action needed."
    return RECOMMENDATIONS.get(root_cause, "Flag this case for manual review by the team.")


def _compute_time_range(db) -> tuple:
    """Anchor a few hours after the latest existing timestamp, and spread forward to
    the real current time - or across a fixed fallback window if the system clock is
    still behind that anchor (can happen since batch 1's jitter could land in what was,
    at the time, the near future)."""
    db_max = db.query(func.max(EvaluationRecord.created_at)).scalar()
    now = datetime.utcnow()

    if db_max is None:
        anchor_start = now - timedelta(hours=1)
    else:
        anchor_start = db_max + timedelta(hours=ANCHOR_OFFSET_HOURS)

    if now > anchor_start:
        span_end = now
    else:
        span_end = anchor_start + timedelta(hours=FALLBACK_SPAN_HOURS)

    return anchor_start, span_end


def _staggered_timestamp(position: int, total: int, start: datetime, end: datetime) -> datetime:
    """Spreads `position` (0..total-1) linearly across [start, end], with jitter so
    records don't land exactly on the interval boundaries."""
    span_seconds = (end - start).total_seconds()
    fraction = position / max(total - 1, 1)
    base = start + timedelta(seconds=span_seconds * fraction)
    jitter = timedelta(minutes=random.uniform(-20, 20))
    return base + jitter


def load_scenarios() -> list:
    with open(SCENARIOS_PATH) as f:
        data = json.load(f)
    return data["scenarios"]


def seed(scenarios: list) -> None:
    # Shuffle so pass/fail and model don't cluster by the JSON file's domain-grouped
    # order - gives more realistic-looking score trends over the seeded window.
    ordered = list(scenarios)
    random.shuffle(ordered)
    total = len(ordered)

    init_db()
    db = SessionLocal()
    try:
        start, end = _compute_time_range(db)

        for position, scenario in enumerate(ordered):
            record = EvaluationRecord(
                project_id=scenario["project_id"],
                question=scenario["question"],
                answer=scenario["answer"],
                model_name=scenario.get("model_name"),
                correctness_score=scenario["expected_correctness_score"],
                faithfulness_score=scenario["expected_faithfulness_score"],
                hallucination_risk=scenario["expected_hallucination_risk"],
                status=scenario["expected_status"],
                explanation=scenario["explanation"],
                latency_ms=round(random.uniform(350.0, 2400.0), 1),
                root_cause=scenario["root_cause"],
                recommendation=_recommendation_for(scenario["root_cause"]),
                evaluator_version=EVALUATOR_VERSION,
                created_at=_staggered_timestamp(position, total, start, end),
            )
            db.add(record)

        db.commit()
    finally:
        db.close()

    print(f"Inserted {total} evaluation records tagged evaluator_version={EVALUATOR_VERSION!r}.")
    print(f"Timestamps spread from {start.isoformat()} to {end.isoformat()}.")


def main() -> None:
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios from {SCENARIOS_PATH}")
    seed(scenarios)


if __name__ == "__main__":
    main()
