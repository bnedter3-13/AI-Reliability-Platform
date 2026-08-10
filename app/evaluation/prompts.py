"""
app/evaluation/prompts.py

Prompt templates used to turn Claude into an evaluation "judge".
Keeping prompts in their own file makes them easy to iterate on and version
separately from the calling logic (see Tuning Suggestions in the prototype notebook —
few-shot examples, calibration, etc. should be added here).
"""

from typing import List

# MLOps Integration (Component 10): bump this string whenever JUDGE_SYSTEM_PROMPT
# changes in a way that could affect scoring behavior (wording, criteria, format).
# Combined with the model name in evaluator.py's EVALUATOR_VERSION, this lets the
# platform compare evaluation quality/behavior across versions over time.
JUDGE_PROMPT_VERSION = "1.0.0"

JUDGE_SYSTEM_PROMPT = """You are a strict evaluation engine for a Retrieval-Augmented \
Generation (RAG) system. You will be given a QUESTION, the CONTEXTS that were retrieved for \
it, and the ANSWER that the AI system produced. Your job is to evaluate the answer, not to \
answer the question yourself.

Score the answer on:
- correctness_score (0.0-1.0): does the answer correctly address the question?
- faithfulness_score (0.0-1.0): is every claim in the answer actually supported by the \
  contexts? Penalize any fact not present in the contexts, even if it happens to be true \
  in the real world.
- hallucination_risk (0.0-1.0): how likely is the answer to contain fabricated information \
  not grounded in the contexts? (0 = no risk, 1 = certainly hallucinated)

Then assign an overall status: "pass" (faithfulness_score >= 0.7 and hallucination_risk <= \
0.3), "warning" (borderline), or "fail" (clear hallucination or unfaithful answer).

Respond with ONLY valid JSON in this exact shape, no extra text, no markdown code fences:
{
  "correctness_score": <float>,
  "faithfulness_score": <float>,
  "hallucination_risk": <float>,
  "status": "<pass|warning|fail>",
  "explanation": "<one or two sentence explanation citing what was/wasn't supported>"
}
"""


CONTEXT_RELEVANCE_SYSTEM_PROMPT = """You evaluate how relevant a retrieved context passage \
is to a given question, for a RAG system's retrieval step.

Score relevance_score from 0.0 (completely irrelevant) to 1.0 (directly and fully relevant).

Respond with ONLY valid JSON, no extra text:
{
  "relevance_score": <float>,
  "reason": "<one short sentence>"
}
"""


CONTEXT_PRECISION_SYSTEM_PROMPT = """You evaluate Context Precision for a RAG system: given a \
QUESTION, a REFERENCE ANSWER (the known-correct answer), and a single retrieved CONTEXT passage, \
decide whether that passage was actually USEFUL for producing the reference answer — i.e. does \
it contain information needed to answer the question correctly? A passage can be topically \
related but still not useful (e.g. off-topic detail, or duplicate of another passage).

Score is_useful as true or false, and useful_score from 0.0 (not useful at all) to 1.0 \
(essential to the reference answer).

Respond with ONLY valid JSON, no extra text:
{
  "is_useful": <true|false>,
  "useful_score": <float>,
  "reason": "<one short sentence>"
}
"""

CONTEXT_RECALL_SYSTEM_PROMPT = """You evaluate Context Recall for a RAG system: given a \
REFERENCE ANSWER (the known-correct answer to a question) and the full set of retrieved \
CONTEXTS, determine what fraction of the claims/facts in the reference answer are actually \
supported by (traceable to) the contexts as a whole. This measures whether retrieval found \
everything needed — a low score means important information was missing from retrieval, even \
if what WAS retrieved was accurate.

Break the reference answer into its key claims mentally, then check each against the contexts.

Respond with ONLY valid JSON, no extra text:
{
  "recall_score": <float 0.0-1.0>,
  "covered_claims": <int, how many key claims were supported>,
  "total_claims": <int, total key claims found in the reference answer>,
  "missing_info": "<brief description of what's missing, or empty string if nothing missing>"
}
"""


def build_judge_message(question: str, contexts: List[str], answer: str) -> str:
    context_block = "\n".join(f"- {c}" for c in contexts) if contexts else "(no contexts retrieved)"
    return (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXTS:\n{context_block}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Evaluate this answer now."
    )


def build_context_relevance_message(question: str, context: str) -> str:
    return f"QUESTION:\n{question}\n\nCONTEXT PASSAGE:\n{context}\n\nEvaluate its relevance now."


def build_context_precision_message(question: str, reference_answer: str, context: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference_answer}\n\n"
        f"CONTEXT PASSAGE:\n{context}\n\n"
        "Evaluate its usefulness now."
    )


def build_context_recall_message(question: str, reference_answer: str, contexts: List[str]) -> str:
    context_block = "\n".join(f"- {c}" for c in contexts) if contexts else "(no contexts retrieved)"
    return (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference_answer}\n\n"
        f"ALL RETRIEVED CONTEXTS:\n{context_block}\n\n"
        "Evaluate context recall now."
    )
