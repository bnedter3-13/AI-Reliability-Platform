"""
tests/test_model_comparator.py

Unit tests for Model Comparison (Component 8). Like test_evaluator.py, these avoid
calling real provider APIs (no network, no cost) by monkeypatching the per-model
generation step (_generate_answer) and the judge (evaluate_answer), and by testing
the pure-logic pieces (message building, cost estimation) directly.
"""

import pytest

import app.comparison.model_comparator as model_comparator
from app.comparison.model_comparator import (
    AVAILABLE_MODELS,
    _build_generation_message,
    _estimate_cost,
    _generate_answer,
    compare_models,
)
from app.evaluation.evaluator import EvaluationResult


def test_build_generation_message_no_contexts():
    message = _build_generation_message("What is 2+2?", [])
    assert message == "Question: What is 2+2?"


def test_build_generation_message_with_contexts():
    message = _build_generation_message("What is 2+2?", ["2+2 equals 4."])
    assert "Contexts:" in message
    assert "- 2+2 equals 4." in message
    assert "Question: What is 2+2?" in message


def test_estimate_cost_known_model():
    cost = _estimate_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.15 + 0.60)


def test_estimate_cost_unknown_model_returns_zero():
    assert _estimate_cost("not-a-real-model", input_tokens=1000, output_tokens=1000) == 0.0


def test_estimate_cost_zero_tokens():
    assert _estimate_cost("gpt-4o", input_tokens=0, output_tokens=0) == 0.0


def test_generate_answer_unknown_model_raises():
    with pytest.raises(ValueError):
        _generate_answer("not-a-real-model", "Q?", [])


def test_compare_models_success(monkeypatch):
    monkeypatch.setattr(
        model_comparator, "_generate_answer",
        lambda model_id, question, contexts: ("a generated answer", 123.4, 10, 20),
    )
    monkeypatch.setattr(
        model_comparator, "evaluate_answer",
        lambda answer, contexts, reference_answer, question: EvaluationResult(
            correctness_score=0.9, faithfulness_score=0.8, hallucination_risk=0.1,
            status="pass", explanation="ok", latency_ms=50.0,
        ),
    )

    results = compare_models(question="Q?", contexts=["ctx"], model_ids=["gpt-4o-mini"])

    assert len(results) == 1
    result = results[0]
    assert result.model_id == "gpt-4o-mini"
    assert result.provider == "openai"
    assert result.answer == "a generated answer"
    assert result.status == "pass"
    assert result.correctness_score == 0.9
    assert result.error is None
    assert result.estimated_cost_usd > 0


def test_compare_models_generation_failure_sets_error(monkeypatch):
    def fake_generate(model_id, question, contexts):
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

    monkeypatch.setattr(model_comparator, "_generate_answer", fake_generate)

    results = compare_models(question="Q?", model_ids=["claude-sonnet-5"])

    assert len(results) == 1
    result = results[0]
    assert result.error == "ANTHROPIC_API_KEY is not configured."
    assert result.answer == ""
    assert result.status is None
    assert result.correctness_score is None


def test_compare_models_judge_failure_leaves_scores_none(monkeypatch):
    monkeypatch.setattr(
        model_comparator, "_generate_answer",
        lambda model_id, question, contexts: ("answer text", 10.0, 5, 5),
    )

    def fake_evaluate(answer, contexts, reference_answer, question):
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(model_comparator, "evaluate_answer", fake_evaluate)

    results = compare_models(question="Q?", model_ids=["gpt-4o"])

    result = results[0]
    assert result.error is None
    assert result.answer == "answer text"
    assert result.status is None
    assert result.correctness_score is None
    assert result.faithfulness_score is None
    assert result.hallucination_risk is None


def test_compare_models_continues_after_one_model_fails(monkeypatch):
    def fake_generate(model_id, question, contexts):
        if model_id == "gpt-4o":
            raise RuntimeError("boom")
        return ("ok answer", 1.0, 1, 1)

    monkeypatch.setattr(model_comparator, "_generate_answer", fake_generate)
    monkeypatch.setattr(
        model_comparator, "evaluate_answer",
        lambda answer, contexts, reference_answer, question: EvaluationResult(
            correctness_score=1.0, faithfulness_score=1.0, hallucination_risk=0.0,
            status="pass", explanation="ok", latency_ms=1.0,
        ),
    )

    results = compare_models(question="Q?", model_ids=["gpt-4o", "gpt-4o-mini"])

    assert len(results) == 2
    assert results[0].error == "boom"
    assert results[1].error is None
    assert results[1].status == "pass"


def test_compare_models_retries_once_after_fail_and_succeeds(monkeypatch):
    generate_calls = []

    def fake_generate(model_id, question, contexts, previous_answer=None):
        generate_calls.append(previous_answer)
        if previous_answer is None:
            return ("first answer", 100.0, 10, 20)
        return ("retried answer", 50.0, 5, 8)

    monkeypatch.setattr(model_comparator, "_generate_answer", fake_generate)

    eval_calls = []

    def fake_evaluate(answer, contexts, reference_answer, question):
        eval_calls.append(answer)
        if answer == "first answer":
            return EvaluationResult(
                correctness_score=0.3, faithfulness_score=0.2, hallucination_risk=0.8,
                status="fail", explanation="not grounded in context", latency_ms=10.0,
            )
        return EvaluationResult(
            correctness_score=0.9, faithfulness_score=0.95, hallucination_risk=0.05,
            status="pass", explanation="now fully grounded", latency_ms=10.0,
        )

    monkeypatch.setattr(model_comparator, "evaluate_answer", fake_evaluate)

    results = compare_models(question="Q?", contexts=["ctx"], model_ids=["gpt-4o-mini"])

    assert len(results) == 1
    result = results[0]
    # Exactly one retry: generation called twice (first with no prior answer,
    # then with the first attempt's answer as previous_answer).
    assert generate_calls == [None, "first answer"]
    assert eval_calls == ["first answer", "retried answer"]
    assert result.answer == "retried answer"
    assert result.status == "pass"
    assert result.correctness_score == 0.9
    assert result.faithfulness_score == 0.95
    assert result.hallucination_risk == 0.05
    assert result.explanation == "[Corrected after 1 retry] now fully grounded"
    # Latency/tokens/cost reflect the sum of both attempts, not just the retry.
    assert result.generation_latency_ms == pytest.approx(150.0, abs=0.1)
    assert result.input_tokens == 15
    assert result.output_tokens == 28
    assert result.estimated_cost_usd == _estimate_cost("gpt-4o-mini", 15, 28)
    assert result.error is None


def test_compare_models_fails_first_retry_passes_second(monkeypatch):
    generate_calls = []

    def fake_generate(model_id, question, contexts, previous_answer=None, corrective_message=None):
        generate_calls.append((previous_answer, corrective_message))
        if previous_answer is None:
            return ("first answer", 100.0, 10, 20)
        if previous_answer == "first answer":
            return ("second answer", 50.0, 5, 8)
        return ("third answer", 30.0, 3, 5)

    monkeypatch.setattr(model_comparator, "_generate_answer", fake_generate)

    eval_calls = []

    def fake_evaluate(answer, contexts, reference_answer, question):
        eval_calls.append(answer)
        if answer in ("first answer", "second answer"):
            return EvaluationResult(
                correctness_score=0.3, faithfulness_score=0.2, hallucination_risk=0.8,
                status="fail", explanation=f"{answer}: not grounded", latency_ms=10.0,
            )
        return EvaluationResult(
            correctness_score=0.9, faithfulness_score=0.95, hallucination_risk=0.05,
            status="pass", explanation="now fully grounded", latency_ms=10.0,
        )

    monkeypatch.setattr(model_comparator, "evaluate_answer", fake_evaluate)

    results = compare_models(question="Q?", contexts=["ctx"], model_ids=["gpt-4o-mini"])

    assert len(results) == 1
    result = results[0]
    # Three generation calls: initial, first retry (default corrective message),
    # second retry (the more pointed, claim-by-claim corrective message).
    assert [c[0] for c in generate_calls] == [None, "first answer", "second answer"]
    assert generate_calls[2][1] == model_comparator.RETRY_CORRECTIVE_MESSAGE_2
    assert eval_calls == ["first answer", "second answer", "third answer"]
    assert result.answer == "third answer"
    assert result.status == "pass"
    assert result.correctness_score == 0.9
    assert result.faithfulness_score == 0.95
    assert result.hallucination_risk == 0.05
    assert result.explanation == "[Corrected after 2 retries] now fully grounded"
    # Latency/tokens/cost reflect the sum of all three attempts.
    assert result.generation_latency_ms == pytest.approx(180.0, abs=0.1)
    assert result.input_tokens == 18
    assert result.output_tokens == 33
    assert result.estimated_cost_usd == _estimate_cost("gpt-4o-mini", 18, 33)
    assert result.error is None


def test_compare_models_fails_both_retries(monkeypatch):
    generate_calls = []

    def fake_generate(model_id, question, contexts, previous_answer=None, corrective_message=None):
        generate_calls.append((previous_answer, corrective_message))
        if previous_answer is None:
            return ("first answer", 100.0, 10, 20)
        if previous_answer == "first answer":
            return ("second answer", 60.0, 6, 9)
        return ("third answer", 40.0, 4, 7)

    monkeypatch.setattr(model_comparator, "_generate_answer", fake_generate)

    def fake_evaluate(answer, contexts, reference_answer, question):
        return EvaluationResult(
            correctness_score=0.2, faithfulness_score=0.1, hallucination_risk=0.9,
            status="fail", explanation=f"{answer}: still ungrounded", latency_ms=10.0,
        )

    monkeypatch.setattr(model_comparator, "evaluate_answer", fake_evaluate)

    results = compare_models(question="Q?", model_ids=["claude-sonnet-5"])

    assert len(results) == 1
    result = results[0]
    # Both retries attempted (three generation calls total); no third retry is
    # made after the second one fails too.
    assert [c[0] for c in generate_calls] == [None, "first answer", "second answer"]
    assert generate_calls[2][1] == model_comparator.RETRY_CORRECTIVE_MESSAGE_2
    assert result.answer == "third answer"
    assert result.status == "fail"
    assert result.explanation == "[Retry attempted twice, still failing] third answer: still ungrounded"
    assert result.generation_latency_ms == pytest.approx(200.0, abs=0.1)
    assert result.input_tokens == 20
    assert result.output_tokens == 36


def test_compare_models_second_retry_evaluation_crash_notes_retry_attempted(monkeypatch):
    generate_calls = []

    def fake_generate(model_id, question, contexts, previous_answer=None, corrective_message=None):
        generate_calls.append(previous_answer)
        if previous_answer is None:
            return ("first answer", 100.0, 10, 20)
        if previous_answer == "first answer":
            return ("second answer", 50.0, 5, 8)
        return ("third answer", 30.0, 3, 5)

    monkeypatch.setattr(model_comparator, "_generate_answer", fake_generate)

    def fake_evaluate(answer, contexts, reference_answer, question):
        if answer in ("first answer", "second answer"):
            return EvaluationResult(
                correctness_score=0.3, faithfulness_score=0.2, hallucination_risk=0.8,
                status="fail", explanation="not grounded", latency_ms=10.0,
            )
        # Judge crashes re-scoring the *second* retry specifically.
        raise RuntimeError("judge exploded on second retry")

    monkeypatch.setattr(model_comparator, "evaluate_answer", fake_evaluate)

    results = compare_models(question="Q?", model_ids=["claude-haiku-4-5-20251001"])

    result = results[0]
    assert generate_calls == [None, "first answer", "second answer"]
    assert result.answer == "third answer"
    assert result.status is None
    assert result.correctness_score is None
    assert result.explanation == (
        "[Retry attempted, evaluation failed] Could not re-score the "
        "retried answer - see server logs."
    )
    assert result.error is None
    assert result.generation_latency_ms == pytest.approx(180.0, abs=0.1)
    assert result.input_tokens == 18
    assert result.output_tokens == 33


def test_compare_models_retry_evaluation_crash_notes_retry_attempted(monkeypatch):
    generate_calls = []

    def fake_generate(model_id, question, contexts, previous_answer=None):
        generate_calls.append(previous_answer)
        if previous_answer is None:
            return ("first answer", 100.0, 10, 20)
        return ("retried answer", 40.0, 4, 6)

    monkeypatch.setattr(model_comparator, "_generate_answer", fake_generate)

    def fake_evaluate(answer, contexts, reference_answer, question):
        if answer == "first answer":
            return EvaluationResult(
                correctness_score=0.3, faithfulness_score=0.2, hallucination_risk=0.8,
                status="fail", explanation="not grounded", latency_ms=10.0,
            )
        # Simulates evaluate_answer's own internal fail-safe not saving it here -
        # e.g. a judge-JSON shape bug, or any other exception during re-evaluation.
        raise RuntimeError("judge exploded on retry")

    monkeypatch.setattr(model_comparator, "evaluate_answer", fake_evaluate)

    results = compare_models(question="Q?", model_ids=["claude-haiku-4-5-20251001"])

    result = results[0]
    # Retry generation still ran and its answer is kept, even though it couldn't
    # be re-scored - and that fact is visible in explanation, not silently blanked.
    assert generate_calls == [None, "first answer"]
    assert result.answer == "retried answer"
    assert result.status is None
    assert result.correctness_score is None
    assert result.faithfulness_score is None
    assert result.hallucination_risk is None
    assert result.explanation == (
        "[Retry attempted, evaluation failed] Could not re-score the "
        "retried answer - see server logs."
    )
    assert result.error is None
    assert result.generation_latency_ms == pytest.approx(140.0, abs=0.1)
    assert result.input_tokens == 14
    assert result.output_tokens == 26


def test_compare_models_no_retry_when_status_is_not_fail(monkeypatch):
    generate_calls = []

    def fake_generate(model_id, question, contexts, previous_answer=None):
        generate_calls.append(previous_answer)
        return ("an answer", 10.0, 1, 1)

    monkeypatch.setattr(model_comparator, "_generate_answer", fake_generate)
    monkeypatch.setattr(
        model_comparator, "evaluate_answer",
        lambda answer, contexts, reference_answer, question: EvaluationResult(
            correctness_score=0.6, faithfulness_score=0.6, hallucination_risk=0.4,
            status="warning", explanation="borderline", latency_ms=5.0,
        ),
    )

    results = compare_models(question="Q?", model_ids=["gpt-4o"])

    assert generate_calls == [None]  # no retry triggered for "warning"
    assert results[0].status == "warning"
    assert results[0].explanation == "borderline"


def test_compare_models_empty_model_ids_defaults_to_all_available(monkeypatch):
    monkeypatch.setattr(
        model_comparator, "_generate_answer",
        lambda model_id, question, contexts: ("ans", 1.0, 1, 1),
    )
    monkeypatch.setattr(
        model_comparator, "evaluate_answer",
        lambda answer, contexts, reference_answer, question: EvaluationResult(
            correctness_score=1.0, faithfulness_score=1.0, hallucination_risk=0.0,
            status="pass", explanation="ok", latency_ms=1.0,
        ),
    )

    # model_ids=[] is falsy, so compare_models() should fall back to every
    # model in AVAILABLE_MODELS rather than running zero comparisons.
    results = compare_models(question="Q?", model_ids=[])

    assert len(results) == len(AVAILABLE_MODELS)
