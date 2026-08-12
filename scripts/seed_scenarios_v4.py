"""
scripts/seed_scenarios_v4.py

Same insert-only pattern as scripts/seed_scenarios_v3.py, loading
data/seed_scenarios_v4.json instead. This batch reflects the retry-with-
clarification mechanism added to compare_models() (app/comparison/model_comparator.py) -
a code-level pipeline change, not a prompt wording change. GENERATION_SYSTEM_PROMPT and
JUDGE_SYSTEM_PROMPT are unchanged from batch 3, so this is tracked by a new, independent
COMPARATOR_VERSION constant instead of bumping GENERATION_PROMPT_VERSION.

`evaluator_version` is pulled live from EVALUATOR_VERSION in app.evaluation.evaluator
(now "claude-sonnet-5:gen1.1.0:judge1.1.0:cmp1.0.0" following the new COMPARATOR_VERSION
constant in app/evaluation/prompts.py, folded in as EVALUATOR_VERSION's third "cmp"
component) - distinct from data/seed_scenarios_v3.json's
"claude-sonnet-5:gen1.1.0:judge1.1.0" (predates COMPARATOR_VERSION) and independently
comparable going forward if the retry logic changes again without a prompt change, or
vice versa.

Timestamps: same anchoring approach as v2/v3 - anchors ANCHOR_OFFSET_HOURS after the
current max created_at in the evaluations table (which now includes batch 3's rows), and
spreads forward either to the real current time or across a fixed fallback window - see
_compute_time_range().

Usage:
    python scripts/seed_scenarios_v4.py
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

SCENARIOS_PATH = Path(__file__).resolve().parent.parent / "data" / "seed_scenarios_v4.json"
ANCHOR_OFFSET_HOURS = 3    # start this many hours after the existing data's latest timestamp
FALLBACK_SPAN_HOURS = 30   # spread window used if the real clock hasn't caught up to the anchor yet


def _recommendation_for(root_cause: str) -> str:
    if root_cause == "none":
        return "No action needed."
    return RECOMMENDATIONS.get(root_cause, "Flag this case for manual review by the team.")


def _compute_time_range(db) -> tuple:
    """Anchor a few hours after the latest existing timestamp, and spread forward to
    the real current time - or across a fixed fallback window if the system clock is
    still behind that anchor (can happen since earlier batches' jitter could land in
    what was, at the time, the near future)."""
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
