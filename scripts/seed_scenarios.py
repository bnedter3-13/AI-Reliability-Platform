"""
scripts/seed_scenarios.py

Loads data/seed_scenarios.json and inserts each scenario as an EvaluationRecord,
using the same DB session/model machinery the real app uses (app.database.connection,
app.database.models) rather than raw SQL, so the seeded rows behave exactly like
rows produced by POST /api/v1/health-checks.

`recommendation` is derived from the real RECOMMENDATIONS dict in
app.root_cause.analyzer (keyed by root_cause) instead of being duplicated in the
JSON file. `evaluator_version` uses the real EVALUATOR_VERSION constant from
app.evaluation.evaluator.

Note: EvaluationRecord has no `context` column (contexts are only used at
evaluation time in the real pipeline, never persisted) - each scenario's `context`
field is read from the JSON for logging only and is not written to the database.

This is an insert-only seed script: running it multiple times adds duplicate rows
rather than upserting, since EvaluationRecord has no natural unique key to dedupe on.

Usage:
    python scripts/seed_scenarios.py
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.connection import SessionLocal, init_db
from app.database.models import EvaluationRecord
from app.evaluation.evaluator import EVALUATOR_VERSION
from app.root_cause.analyzer import RECOMMENDATIONS

SCENARIOS_PATH = Path(__file__).resolve().parent.parent / "data" / "seed_scenarios.json"
DAYS_SPAN = 6  # scenarios are staggered across roughly this many past days


def _recommendation_for(root_cause: str) -> str:
    if root_cause == "none":
        return "No action needed."
    return RECOMMENDATIONS.get(root_cause, "Flag this case for manual review by the team.")


def _staggered_timestamp(position: int, total: int) -> datetime:
    """Spreads `position` (0..total-1) across the past DAYS_SPAN days, oldest first,
    with hour/minute jitter so records within the same day don't land on the hour."""
    now = datetime.utcnow()
    days_ago = DAYS_SPAN * (total - position) / total
    jitter = timedelta(hours=random.uniform(0, 20), minutes=random.uniform(0, 59))
    return now - timedelta(days=days_ago) + jitter


def load_scenarios() -> list:
    with open(SCENARIOS_PATH) as f:
        data = json.load(f)
    return data["scenarios"]


def seed(scenarios: list) -> None:
    # Shuffle so scenario categories (pass/fail, domain) are mixed across the
    # timeline instead of clustering by their order in the JSON file (which is
    # grouped by domain) - gives more realistic-looking score trends.
    ordered = list(scenarios)
    random.shuffle(ordered)
    total = len(ordered)

    init_db()
    db = SessionLocal()
    try:
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
                created_at=_staggered_timestamp(position, total),
            )
            db.add(record)

        db.commit()
    finally:
        db.close()

    print(f"Inserted {total} evaluation records across the past {DAYS_SPAN} days.")


def main() -> None:
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios from {SCENARIOS_PATH}")
    seed(scenarios)


if __name__ == "__main__":
    main()
