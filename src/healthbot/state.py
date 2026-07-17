from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RetrievalDecision(BaseModel):
    use_local_only: bool
    needs_web_fallback: bool
    reason: str


class Quiz(BaseModel):
    question: str
    expected_answer: str
    supporting_claim: str


class QuizEvaluation(BaseModel):
    grade: Literal["correct", "partially_correct", "incorrect"]
    explanation: str
    correct_answer: str
    citations: list[str] = Field(default_factory=list)


class SafetyReview(BaseModel):
    safe: bool
    grounded: bool
    escalation_needed: bool
    rationale: str
    revised_response: str | None = None


class State(MessagesState):
    messages: Annotated[list[AnyMessage], add_messages]
    workflow_stage: str
    topic: str | None
    question: str | None
    documents: list[Document]
    retrieved_context: str | None
    source_citations: list[str]
    summary: str | None
    quiz: Quiz | None
    patient_answer: str | None
    evaluation: QuizEvaluation | None
    feedback: str | None
    retrieval_decision: RetrievalDecision | None
    retrieval_scores: list[float]
    used_web_fallback: bool
    safety_review: SafetyReview | None
