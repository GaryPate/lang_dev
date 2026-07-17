from __future__ import annotations

from typing import Any

from langfuse import get_client


def get_langfuse_client():
    try:
        return get_client()
    except Exception:
        return None


def trace_retrieval(
    question: str, documents: list[dict], scores: list[float], decision: str
) -> None:
    langfuse = get_langfuse_client()
    if not langfuse:
        return

    with langfuse.start_as_current_observation(
        as_type="span",
        name="retrieve_local_docs",
        input={"question": question},
    ) as span:
        span.update(
            output={
                "documents": documents,
                "scores": scores,
                "decision": decision,
            }
        )


def trace_generation(
    question: str,
    context: str,
    answer: str,
    citations: list[str],
) -> None:
    langfuse = get_langfuse_client()
    if not langfuse:
        return

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="summarise_context",
        input={
            "question": question,
            "context_preview": context[:1500],
            "citation_candidates": citations,
        },
    ) as generation:
        generation.update(
            output={
                "answer": answer,
                "citation_count": len(citations),
            }
        )


def trace_web_fallback(question: str, urls: list[str], result_count: int) -> None:
    langfuse = get_langfuse_client()
    if not langfuse:
        return

    with langfuse.start_as_current_observation(
        as_type="span",
        name="search_web_fallback",
        input={"question": question},
    ) as span:
        span.update(
            output={
                "result_count": result_count,
                "urls": urls,
            }
        )


def trace_quiz_evaluation(
    question: str,
    user_answer: str,
    grade: str,
    citations: list[str],
) -> None:
    langfuse = get_langfuse_client()
    if not langfuse:
        return

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="evaluate_answer",
        input={
            "question": question,
            "user_answer": user_answer,
        },
    ) as generation:
        generation.update(
            output={
                "grade": grade,
                "citations": citations,
            }
        )


def trace_safety_review(
    question: str,
    safe: bool,
    grounded: bool,
    escalation_needed: bool,
    rationale: str,
) -> None:
    langfuse = get_langfuse_client()
    if not langfuse:
        return

    with langfuse.start_as_current_observation(
        as_type="span",
        name="review_safety",
        input={"question": question},
    ) as span:
        span.update(
            output={
                "safe": safe,
                "grounded": grounded,
                "escalation_needed": escalation_needed,
                "rationale": rationale,
            }
        )


def flush_langfuse() -> None:
    langfuse = get_langfuse_client()
    if langfuse:
        langfuse.flush()
