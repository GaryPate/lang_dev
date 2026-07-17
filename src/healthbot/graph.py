import os
import warnings
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from tavily import TavilyClient

from .config import settings
from .models import build_chat_llm
from .observability import (
    trace_generation,
    trace_quiz_evaluation,
    trace_retrieval,
    trace_safety_review,
    trace_web_fallback,
)
from .prompts import (
    EVALUATION_SYSTEM_PROMPT,
    GLOBAL_SYSTEM_PROMPT,
    QUIZ_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
)
from .retrieval import decide_retrieval_path, format_context, retrieve_with_scores
from .safety import run_safety_review
from .state import Quiz, QuizEvaluation, State

warnings.filterwarnings(
    "ignore",
    message=".*Deserializing unregistered type.*",
)

load_dotenv()


def entry_point(state: State, config: RunnableConfig):
    return {"workflow_stage": "requesting_topic"}


def ask_patient(state: State):
    topic = interrupt(
        {
            "type": "request_topic",
            "message": "What health topic or medical condition would you like to learn about?",
        }
    )
    return {
        "topic": str(topic),
        "question": str(topic),
        "workflow_stage": "topic_received",
    }


def retrieve_local_docs(state: State):
    query = state["topic"] or state["question"] or ""
    results = retrieve_with_scores(query, k=4)
    documents = [doc for doc, _ in results]
    scores = [score for _, score in results]
    retrieved_context, citations = format_context(documents)

    return {
        "documents": documents,
        "retrieved_context": retrieved_context,
        "source_citations": citations,
        "retrieval_scores": scores,
        "workflow_stage": "local_docs_retrieved",
    }


def grade_retrieval(state: State):
    paired = list(zip(state["documents"], state["retrieval_scores"]))
    decision = decide_retrieval_path(paired)

    trace_retrieval(
        question=state["question"] or "",
        documents=[
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "filename": doc.metadata.get("filename"),
                "source_name": doc.metadata.get("source_name"),
                "source_url": doc.metadata.get("source_url"),
            }
            for doc in state["documents"]
        ],
        scores=state["retrieval_scores"],
        decision=decision.reason,
    )

    return {
        "retrieval_decision": decision,
        "workflow_stage": "retrieval_graded",
    }


def search_web_fallback(state: State):
    tavily = TavilyClient(api_key=settings.tavily_api_key)
    query = state["topic"] or state["question"] or ""
    results = tavily.search(query=query, max_results=3, search_depth="advanced")

    web_blocks = []
    web_citations = []
    for item in results["results"]:
        web_blocks.append(
            f"[Web Source: {item['title']} | URL: {item['url']}]\n{item['content']}"
        )
        web_citations.append(item["url"])

    combined_context = state["retrieved_context"] or ""
    if web_blocks:
        combined_context = (combined_context + "\n\n" + "\n\n".join(web_blocks)).strip()

    trace_web_fallback(
        question=state["question"] or "",
        urls=web_citations,
        result_count=len(web_citations),
    )

    return {
        "retrieved_context": combined_context,
        "source_citations": list(
            dict.fromkeys(state["source_citations"] + web_citations)
        ),
        "used_web_fallback": True,
        "workflow_stage": "web_fallback_completed",
    }


def summarise_context(state: State):
    llm = build_chat_llm()
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question:\n{state['question']}\n\n"
                f"Context:\n{state['retrieved_context']}\n\n"
                f"Allowed citations:\n{state['source_citations']}"
            )
        ),
    ]

    ai_message = llm.invoke(messages)

    trace_generation(
        question=state["question"] or "",
        context=state["retrieved_context"] or "",
        answer=ai_message.content,
        citations=state["source_citations"],
    )

    return {
        "messages": [ai_message],
        "summary": ai_message.content,
        "workflow_stage": "summary_ready",
    }


def quiz_wait(state: State):
    """
    Provides the summary from the state and prompts the user when ready for the quiz.
    """

    summary_text = state["messages"][-1].content

    ready = interrupt(
        {
            "type": "confirmation",
            "message1": summary_text,
            "message2": "Are you ready for the quiz?",
            "options": ["Y", "N"],
        }
    )

    return {"quiz_ready": bool(ready)}


def generate_quiz(state: State):
    """
    Obtains a structured quiz from the LLM by feeding the topic and summary.
    """

    quiz_model = build_chat_llm().with_structured_output(Quiz)

    summary = state["summary"]

    messages = [
        SystemMessage(content=GLOBAL_SYSTEM_PROMPT + "\n" + QUIZ_SYSTEM_PROMPT),
        HumanMessage(content=(f"Topic:\n{state['topic']}\nSummary:\n{summary}")),
    ]

    ai_response = quiz_model.invoke(messages)

    return {"quiz": ai_response}


def quiz_answer(state: State):
    """
    Provides the quiz to the user and awaits an answer.
    """
    answer = interrupt(
        {
            "type": "quiz_answer",
            "message": state["quiz"].question,
        }
    )

    return {
        "patient_answer": answer,
        "workflow_stage": "answer_received",
    }


def format_evaluation(evaluation: QuizEvaluation) -> str:
    output = f"That is {evaluation.grade}. {evaluation.explanation}\n\n"
    if evaluation.grade != "correct":
        output += f"The correct answer is: {evaluation.correct_answer}\n\n"
    output += "Citations:\n" + "\n".join(f"- {c}" for c in evaluation.citations)
    return output


def evaluate_answer(state: State):
    evaluator = build_chat_llm().with_structured_output(QuizEvaluation)
    messages = [
        SystemMessage(content=EVALUATION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Summary:\n{state['summary']}\n\n"
                f"Quiz:\n{state['quiz'].model_dump_json(indent=2)}\n\n"
                f"Answer:\n{state['patient_answer']}\n\n"
                f"Allowed citations:\n{state['source_citations']}"
            )
        ),
    ]
    evaluation = evaluator.invoke(messages)

    trace_quiz_evaluation(
        question=state["question"] or "",
        user_answer=state["patient_answer"] or "",
        grade=evaluation.grade,
        citations=evaluation.citations,
    )

    return {
        "evaluation": evaluation,
        "feedback": format_evaluation(evaluation),
        "workflow_stage": "answer_evaluated",
    }


def review_safety(state: State):
    review = run_safety_review(
        question=state["question"] or "",
        answer_text=state["summary"] or "",
        context=state["retrieved_context"] or "",
        allowed_citations=state["source_citations"],
    )

    trace_safety_review(
        question=state["question"] or "",
        safe=review.safe,
        grounded=review.grounded,
        escalation_needed=review.escalation_needed,
        rationale=review.rationale,
    )

    updated_feedback = state["feedback"]
    if review.revised_response:
        updated_feedback = (
            updated_feedback + "\n\nSafety review note:\n" + review.revised_response
        )

    return {
        "safety_review": review,
        "feedback": updated_feedback,
        "workflow_stage": "safety_review_complete",
    }


def continue_or_exit(state: State) -> Command:
    choice = interrupt(
        {
            "type": "continue_or_exit",
            "feedback": state["feedback"],
            "message": "Would you like to research another health topic?",
            "options": ["Y", "N"],
        }
    )

    if str(choice).strip().lower() in {"y", "yes"}:
        return Command(
            update={
                "topic": None,
                "question": None,
                "documents": [],
                "retrieved_context": None,
                "source_citations": [],
                "summary": None,
                "quiz": None,
                "patient_answer": None,
                "evaluation": None,
                "feedback": None,
                "retrieval_decision": None,
                "retrieval_scores": [],
                "used_web_fallback": False,
                "safety_review": None,
                "workflow_stage": "requesting_topic",
            },
            goto="ask_patient",
        )

    return Command(goto=END)


def retrieval_router(state: State) -> str:
    decision = state["retrieval_decision"]
    if decision and decision.needs_web_fallback:
        return "web"
    return "local"


def build_graph():
    workflow = StateGraph(State)

    workflow.add_node("entry_point", entry_point)
    workflow.add_node("ask_patient", ask_patient)
    workflow.add_node("retrieve_local_docs", retrieve_local_docs)
    workflow.add_node("grade_retrieval", grade_retrieval)
    workflow.add_node("search_web_fallback", search_web_fallback)
    workflow.add_node("summarise_context", summarise_context)
    workflow.add_node("quiz_wait", quiz_wait)
    workflow.add_node("generate_quiz", generate_quiz)
    workflow.add_node("quiz_answer", quiz_answer)
    workflow.add_node("evaluate_answer", evaluate_answer)
    workflow.add_node("review_safety", review_safety)
    workflow.add_node("continue_or_exit", continue_or_exit)

    workflow.add_edge(START, "entry_point")
    workflow.add_edge("entry_point", "ask_patient")
    workflow.add_edge("ask_patient", "retrieve_local_docs")
    workflow.add_edge("retrieve_local_docs", "grade_retrieval")
    workflow.add_conditional_edges(
        "grade_retrieval",
        retrieval_router,
        {
            "local": "summarise_context",
            "web": "search_web_fallback",
        },
    )
    workflow.add_edge("search_web_fallback", "summarise_context")
    workflow.add_edge("summarise_context", "quiz_wait")
    workflow.add_edge("quiz_wait", "generate_quiz")
    workflow.add_edge("generate_quiz", "quiz_answer")
    workflow.add_edge("quiz_answer", "evaluate_answer")
    workflow.add_edge("evaluate_answer", "review_safety")
    workflow.add_edge("review_safety", "continue_or_exit")

    mem_serial = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("healthbot.state", "Quiz"),
            ("healthbot.state", "QuizEvaluation"),
            ("healthbot.state", "RetrievalDecision"),
            ("healthbot.state", "SafetyReview"),
        ]
    )

    return workflow.compile(checkpointer=InMemorySaver(serde=mem_serial))


def initial_state() -> dict:
    return {
        "messages": [],
        "workflow_stage": "started",
        "topic": None,
        "question": None,
        "documents": [],
        "retrieved_context": None,
        "source_citations": [],
        "summary": None,
        "quiz": None,
        "patient_answer": None,
        "evaluation": None,
        "feedback": None,
        "retrieval_decision": None,
        "retrieval_scores": [],
        "used_web_fallback": False,
        "safety_review": None,
    }
