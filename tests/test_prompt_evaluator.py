"""
tests/test_prompt_evaluator.py

Unit tests for Prompt Evaluation (Component 7). Covers the pure-logic pieces
(_extract_json's repair attempts, score/issue coercion) directly, and exercises
evaluate_prompt() by monkeypatching the module-level `_client` with a fake object
so no real Claude API call is made (no network, no cost), matching the approach
in test_evaluator.py.
"""

from types import SimpleNamespace

import pytest

import app.evaluation.prompt_evaluator as prompt_evaluator
from app.evaluation.prompt_evaluator import (
    _coerce_issues,
    _coerce_score,
    _extract_json,
    evaluate_prompt,
)


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

def test_extract_json_plain():
    text = (
        '{"clarity_score": 0.8, "completeness_score": 0.7, '
        '"hallucination_risk_score": 0.2, "issues": [], "suggested_rewrite": ""}'
    )
    result = _extract_json(text)
    assert result["clarity_score"] == 0.8


def test_extract_json_repairs_escaped_apostrophes():
    # LLM-generated JSON sometimes escapes apostrophes as \' inside strings, which
    # is not valid JSON - _extract_json should repair this rather than raise.
    text = (
        "{\"clarity_score\": 0.5, \"completeness_score\": 0.5, "
        "\"hallucination_risk_score\": 0.5, \"issues\": [\"doesn\\'t specify tone\"], "
        "\"suggested_rewrite\": \"\"}"
    )
    result = _extract_json(text)
    assert result["issues"] == ["doesn't specify tone"]


def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError):
        _extract_json("not json at all")


# ---------------------------------------------------------------------------
# _coerce_score / _coerce_issues
# ---------------------------------------------------------------------------

def test_coerce_score_normal_value():
    assert _coerce_score(0.42) == 0.42


def test_coerce_score_clamps_above_one():
    assert _coerce_score(1.5) == 1.0


def test_coerce_score_clamps_below_zero():
    assert _coerce_score(-0.3) == 0.0


def test_coerce_score_invalid_input_defaults_to_zero():
    assert _coerce_score(None) == 0.0
    assert _coerce_score("not-a-number") == 0.0


def test_coerce_issues_list_of_strings():
    assert _coerce_issues(["missing tone guidance", "no fallback"]) == [
        "missing tone guidance", "no fallback",
    ]


def test_coerce_issues_single_string_is_wrapped_in_a_list():
    assert _coerce_issues("only one issue") == ["only one issue"]


def test_coerce_issues_none_returns_empty_list():
    assert _coerce_issues(None) == []


# ---------------------------------------------------------------------------
# evaluate_prompt
# ---------------------------------------------------------------------------

def test_evaluate_prompt_no_client_raises(monkeypatch):
    monkeypatch.setattr(prompt_evaluator, "_client", None)
    with pytest.raises(RuntimeError):
        evaluate_prompt("You are a helpful assistant.")


def test_evaluate_prompt_empty_text_raises(monkeypatch):
    # A truthy dummy client so the empty-text check (not the client check) is
    # what's actually under test here.
    monkeypatch.setattr(prompt_evaluator, "_client", object())
    with pytest.raises(ValueError):
        evaluate_prompt("   ")


def test_evaluate_prompt_success(monkeypatch):
    fake_response_text = (
        '{"clarity_score": 0.9, "completeness_score": 0.6, '
        '"hallucination_risk_score": 0.2, "issues": ["no fallback for missing info"], '
        '"suggested_rewrite": "Improved prompt text."}'
    )

    class _FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text=fake_response_text, type="text")])

    monkeypatch.setattr(prompt_evaluator, "_client", SimpleNamespace(messages=_FakeMessages()))

    result = evaluate_prompt("You are a helpful assistant.")

    assert result.clarity_score == 0.9
    assert result.completeness_score == 0.6
    assert result.hallucination_risk_score == 0.2
    assert result.issues == ["no fallback for missing info"]
    assert result.suggested_rewrite == "Improved prompt text."


def test_evaluate_prompt_coerces_malformed_scores(monkeypatch):
    # Judge returns an out-of-range score and a single string instead of a list
    # for issues - evaluate_prompt should coerce rather than crash.
    fake_response_text = (
        '{"clarity_score": 1.7, "completeness_score": "bad-data", '
        '"hallucination_risk_score": 0.1, "issues": "just one issue", '
        '"suggested_rewrite": ""}'
    )

    class _FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text=fake_response_text, type="text")])

    monkeypatch.setattr(prompt_evaluator, "_client", SimpleNamespace(messages=_FakeMessages()))

    result = evaluate_prompt("Some prompt.")

    assert result.clarity_score == 1.0
    assert result.completeness_score == 0.0
    assert result.issues == ["just one issue"]
