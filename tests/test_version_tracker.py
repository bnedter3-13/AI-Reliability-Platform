"""
tests/test_version_tracker.py

Unit tests for MLOps Integration (Component 10). Uses the in-memory `db_session`
fixture from conftest.py instead of the real database, and covers the aggregation
(list_versions, _summarize_version), comparison (compare_versions), and reporting
(generate_periodic_report) logic with hand-built EvaluationRecord rows.
"""

import datetime

import pytest

from app.database.models import EvaluationRecord
from app.mlops.version_tracker import (
    _summarize_version,
    compare_versions,
    generate_periodic_report,
    list_versions,
)


def _make_record(
    evaluator_version="v1",
    project_id="proj1",
    status="pass",
    faithfulness_score=0.8,
    correctness_score=0.8,
    hallucination_risk=0.1,
    latency_ms=100.0,
    created_at=None,
):
    return EvaluationRecord(
        project_id=project_id,
        question="Q?",
        answer="A.",
        correctness_score=correctness_score,
        faithfulness_score=faithfulness_score,
        hallucination_risk=hallucination_risk,
        status=status,
        latency_ms=latency_ms,
        evaluator_version=evaluator_version,
        created_at=created_at or datetime.datetime.utcnow(),
    )


def test_list_versions_empty(db_session):
    assert list_versions(db_session) == []


def test_list_versions_groups_and_orders_most_recent_first(db_session):
    now = datetime.datetime.utcnow()
    db_session.add_all([
        _make_record(evaluator_version="v1", created_at=now - datetime.timedelta(days=2)),
        _make_record(evaluator_version="v1", created_at=now - datetime.timedelta(days=1)),
        _make_record(evaluator_version="v2", created_at=now),
    ])
    db_session.commit()

    versions = list_versions(db_session)

    assert [v["evaluator_version"] for v in versions] == ["v2", "v1"]
    assert versions[1]["sample_size"] == 2


def test_list_versions_treats_null_version_as_unknown(db_session):
    db_session.add(_make_record(evaluator_version=None))
    db_session.commit()

    versions = list_versions(db_session)

    assert versions[0]["evaluator_version"] == "unknown (pre-versioning)"


def test_list_versions_filters_by_project_id(db_session):
    db_session.add_all([
        _make_record(evaluator_version="v1", project_id="proj1"),
        _make_record(evaluator_version="v2", project_id="proj2"),
    ])
    db_session.commit()

    versions = list_versions(db_session, project_id="proj1")

    assert len(versions) == 1
    assert versions[0]["evaluator_version"] == "v1"


def test_summarize_version_no_data_returns_zeros(db_session):
    summary = _summarize_version(db_session, "v-does-not-exist")

    assert summary.sample_size == 0
    assert summary.avg_faithfulness == 0.0
    assert summary.pass_rate == 0.0


def test_summarize_version_computes_averages_and_pass_rate(db_session):
    db_session.add_all([
        _make_record(evaluator_version="v1", status="pass", faithfulness_score=0.9),
        _make_record(evaluator_version="v1", status="pass", faithfulness_score=0.7),
        _make_record(evaluator_version="v1", status="fail", faithfulness_score=0.2),
    ])
    db_session.commit()

    summary = _summarize_version(db_session, "v1")

    assert summary.sample_size == 3
    assert summary.avg_faithfulness == pytest.approx(0.6, abs=0.001)
    assert summary.pass_rate == pytest.approx(2 / 3, abs=0.001)


def test_compare_versions_not_enough_data(db_session):
    db_session.add(_make_record(evaluator_version="v1"))
    db_session.commit()

    comparison = compare_versions(db_session, "v1", "v2-never-used")

    assert "Not enough data" in comparison.verdict


def test_compare_versions_detects_improvement(db_session):
    db_session.add_all([
        _make_record(evaluator_version="v1", status="pass", faithfulness_score=0.6),
        _make_record(evaluator_version="v2", status="pass", faithfulness_score=0.9),
    ])
    db_session.commit()

    comparison = compare_versions(db_session, "v1", "v2")

    assert comparison.faithfulness_delta == pytest.approx(0.3, abs=0.001)
    assert "improvement" in comparison.verdict


def test_compare_versions_detects_regression(db_session):
    db_session.add_all([
        _make_record(evaluator_version="v1", status="pass", faithfulness_score=0.9),
        _make_record(evaluator_version="v2", status="fail", faithfulness_score=0.5),
    ])
    db_session.commit()

    comparison = compare_versions(db_session, "v1", "v2")

    assert comparison.faithfulness_delta < 0
    assert "WORSE" in comparison.verdict


def test_compare_versions_no_significant_difference(db_session):
    db_session.add_all([
        _make_record(evaluator_version="v1", status="pass", faithfulness_score=0.80),
        _make_record(evaluator_version="v2", status="pass", faithfulness_score=0.81),
    ])
    db_session.commit()

    comparison = compare_versions(db_session, "v1", "v2")

    assert comparison.verdict == "No significant difference detected between the two versions."


def test_generate_periodic_report_no_evaluations(db_session):
    report = generate_periodic_report(db_session)

    assert report["current_version"] is None
    assert report["version_comparison"] is None
    assert report["suggested_action"] == "No evaluations recorded yet."


def test_generate_periodic_report_single_version_has_no_comparison(db_session):
    db_session.add(_make_record(evaluator_version="v1"))
    db_session.commit()

    report = generate_periodic_report(db_session)

    assert report["current_version"] == "v1"
    assert report["version_comparison"] is None
    assert report["current_version_summary"]["sample_size"] == 1


def test_generate_periodic_report_flags_regression(db_session):
    now = datetime.datetime.utcnow()
    db_session.add_all([
        _make_record(evaluator_version="v1", status="pass", faithfulness_score=0.9,
                      created_at=now - datetime.timedelta(days=1)),
        _make_record(evaluator_version="v2", status="fail", faithfulness_score=0.3,
                      created_at=now),
    ])
    db_session.commit()

    report = generate_periodic_report(db_session)

    assert report["current_version"] == "v2"
    assert "reverting" in report["suggested_action"]
