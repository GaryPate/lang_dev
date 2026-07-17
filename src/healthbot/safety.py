from langchain_core.messages import HumanMessage, SystemMessage

from .models import build_chat_llm
from .prompts import SAFETY_REVIEW_PROMPT
from .state import SafetyReview


def run_safety_review(
    question: str,
    answer_text: str,
    context: str,
    allowed_citations: list[str],
) -> SafetyReview:
    reviewer = build_chat_llm().with_structured_output(SafetyReview)
    messages = [
        SystemMessage(content=SAFETY_REVIEW_PROMPT),
        HumanMessage(
            content=(
                f"User question:\n{question}\n\n"
                f"Answer draft:\n{answer_text}\n\n"
                f"Retrieved context:\n{context}\n\n"
                f"Allowed citations:\n{allowed_citations}"
            )
        ),
    ]
    return reviewer.invoke(messages)
