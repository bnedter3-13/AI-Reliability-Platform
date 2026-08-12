"""
tests/test_analysis_agent.py

Unit tests for the AI Analysis Agent (Component 3). Uses the in-memory `db_session`
fixture from conftest.py for the batch-of-evaluations lookup, and monkeypatches the
module-level `_client` with a fake object for the pattern-finding call itself, so no
real Claude API call is made (no network, no cost) - same approach as
test_rag_evaluator.py and test_prompt_evaluator.py.
"""

import datetime
from types import SimpleNamespace

import pytest

import app.agents.analysis_agent as analysis_agent
from app.agents.analysis_agent import (
    _extract_json,
    _format_batch,
    analyze_recent_patterns,
)
from app.database.models import EvaluationRecord


def _make_record(
    question="Q?",
    status="fail",
    faithfulness_score=0.3,
    hallucination_risk=0.7,
    root_cause="hallucination",
    project_id="proj1",
    created_at=None,
):
    return EvaluationRecord(
        project_id=project_id,
        question=question,
        answer="A.",
        correctness_score=0.3,
        faithfulness_score=faithfulness_score,
        hallucination_risk=hallucination_risk,
        status=status,
        root_cause=root_cause,
        created_at=created_at or datetime.datetime.utcnow(),
    )


class _FakeMessages:
    def __init__(self, text):
        self._text = text

    def create(self, **kwargs):
        return SimpleNamespace(content=[SimpleNamespace(text=self._text, type="text")])


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

def test_extract_json_plain():
    text = (
        '{"summary": "ok", "dominant_issue": "none", "pattern_detected": false, '
        '"pattern_description": "", "severity": "low", "suggested_action": "none"}'
    )
    result = _extract_json(text)
    assert result["dominant_issue"] == "none"


def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


def test_extract_json_raises_on_malformed_json():
    # Unlike prompt_evaluator's _extract_json, this one has no repair attempts -
    # malformed JSON should still surface as a ValueError (json.JSONDecodeError
    # is a ValueError subclass), not crash with something unexpected.
    with pytest.raises(ValueError):
        _extract_json('{"summary": "trailing comma",}')


# ---------------------------------------------------------------------------
# _format_batch
# ---------------------------------------------------------------------------

def test_format_batch_includes_all_records():
    records = [
        _make_record(question="Why did it fail?", status="fail", root_cause="hallucination"),
        _make_record(question="Is this correct?", status="pass", root_cause=None),
    ]

    formatted = _format_batch(records)
    lines = formatted.split("\n")

    assert len(lines) == 2
    assert "Why did it fail?" in lines[0]
    assert "root_cause=hallucination" in lines[0]
    assert "root_cause=none" in lines[1]  # None root_cause defaults to "none"


def test_format_batch_empty_list():
    assert _format_batch([]) == ""


# ---------------------------------------------------------------------------
# analyze_recent_patterns
# ---------------------------------------------------------------------------

def test_analyze_recent_patterns_no_records_returns_default_report(db_session):
    report = analyze_recent_patterns(db_session)

    assert report.sample_size == 0
    assert report.summary == "No evaluations recorded yet."
    assert report.pattern_detected is False


def test_analyze_recent_patterns_no_client_raises_when_records_exist(db_session, monkeypatch):
    db_session.add(_make_record())
    db_session.commit()
    monkeypatch.setattr(analysis_agent, "_client", None)

    with pytest.raises(RuntimeError):
        analyze_recent_patterns(db_session)


def test_analyze_recent_patterns_returns_report_from_client(db_session, monkeypatch):
    db_session.add_all([
        _make_record(question="Q1", root_cause="hallucination"),
        _make_record(question="Q2", root_cause="hallucination"),
    ])
    db_session.commit()

    fake_text = (
        '{"summary": "Hallucination keeps recurring.", "dominant_issue": "hallucination", '
        '"pattern_detected": true, "pattern_description": "2 of 2 failures are hallucinations.", '
        '"severity": "high", "suggested_action": "Tighten the judge prompt."}'
    )
    monkeypatch.setattr(analysis_agent, "_client", SimpleNamespace(messages=_FakeMessages(fake_text)))

    report = analyze_recent_patterns(db_session)

    assert report.sample_size == 2
    assert report.dominant_issue == "hallucination"
    assert report.pattern_detected is True
    assert report.severity == "high"


def test_analyze_recent_patterns_respects_limit(db_session, monkeypatch):
    db_session.add_all([_make_record(question=f"Q{i}") for i in range(5)])
    db_session.commit()

    fake_text = (
        '{"summary": "ok", "dominant_issue": "none", "pattern_detected": false, '
        '"pattern_description": "", "severity": "low", "suggested_action": "none"}'
    )
    monkeypatch.setattr(analysis_agent, "_client", SimpleNamespace(messages=_FakeMessages(fake_text)))

    report = analyze_recent_patterns(db_session, limit=3)

    assert report.sample_size == 3


def test_analyze_recent_patterns_filters_by_project_id(db_session, monkeypatch):
    db_session.add_all([
        _make_record(project_id="proj1"),
        _make_record(project_id="proj2"),
    ])
    db_session.commit()

    fake_text = (
        '{"summary": "ok", "dominant_issue": "none", "pattern_detected": false, '
        '"pattern_description": "", "severity": "low", "suggested_action": "none"}'
    )
    monkeypatch.setattr(analysis_agent, "_client", SimpleNamespace(messages=_FakeMessages(fake_text)))

    report = analyze_recent_patterns(db_session, project_id="proj1")

    assert report.sample_size == 1
