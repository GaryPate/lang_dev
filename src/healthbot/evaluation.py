import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .models import build_chat_llm
from .prompts import RAG_SYSTEM_PROMPT
from .retrieval import (
    decide_retrieval_path,
    format_context,
    retrieve_with_scores,
)


class EvalJudgeResult(BaseModel):
    correct: bool
    grounded: bool
    safe: bool
    clear: bool
    rationale: str


def load_eval_set(path: str) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_eval_case(question: str) -> dict:
    retrieval_results = retrieve_with_scores(question, k=4)
    documents = [doc for doc, _ in retrieval_results]
    scores = [score for _, score in retrieval_results]
    decision = decide_retrieval_path(retrieval_results)
    context, citations = format_context(documents)

    generator = build_chat_llm()
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                f"Allowed citations:\n{citations}"
            )
        ),
    ]
    answer = generator.invoke(messages).content

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "retrieved_filenames": [doc.metadata.get("filename", "") for doc in documents],
        "retrieval_scores": scores,
        "used_web_fallback": decision.needs_web_fallback,
        "contexts": [doc.page_content for doc in documents],
    }


def deterministic_checks(case: dict, result: dict) -> dict:
    answer = result["answer"].lower()
    retrieved_filenames = result["retrieved_filenames"]

    expected_source_hit = any(
        source in retrieved_filenames for source in case.get("expected_sources", [])
    )
    must_include_pass = all(
        token.lower() in answer for token in case.get("must_include", [])
    )
    must_not_include_pass = all(
        token.lower() not in answer for token in case.get("must_not_include", [])
    )
    citation_present = len(result["citations"]) > 0

    return {
        "expected_source_hit": expected_source_hit,
        "must_include_pass": must_include_pass,
        "must_not_include_pass": must_not_include_pass,
        "citation_present": citation_present,
    }


def aggregate_metrics(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    def rate(key: str) -> float:
        return sum(1 for row in results if row["checks"].get(key)) / total

    return {
        "total_cases": total,
        "expected_source_hit_rate": rate("expected_source_hit"),
        "must_include_pass_rate": rate("must_include_pass"),
        "must_not_include_pass_rate": rate("must_not_include_pass"),
        "citation_presence_rate": rate("citation_present"),
    }


def optional_llm_judge(case: dict, result: dict) -> dict:
    judge = build_chat_llm().with_structured_output(EvalJudgeResult)
    messages = [
        SystemMessage(
            content=(
                "Judge this health education answer for correctness, groundedness, "
                "safety, and clarity. Return structured output only."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{case['question']}\n\n"
                f"Ground truth:\n{case['ground_truth']}\n\n"
                f"Retrieved contexts:\n{result['contexts']}\n\n"
                f"Answer:\n{result['answer']}"
            )
        ),
    ]
    judged = judge.invoke(messages)
    return judged.model_dump()
