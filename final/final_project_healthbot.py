import os
import warnings
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from tavily import TavilyClient

from null.prompts_healthbot import (
    EVALUATION_SYSTEM_PROMPT,
    GLOBAL_SYSTEM_PROMPT,
    QUIZ_SYSTEM_PROMPT,
    SEARCH_QUERY_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)

warnings.filterwarnings(
    "ignore",
    message=".*Deserializing unregistered type.*",
)

load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)


class Quiz(BaseModel):
    question: str
    expected_answer: str
    supporting_claim: str


class QuizEvaluation(BaseModel):
    grade: Literal["correct", "partially_correct", "incorrect"]
    explanation: str
    correct_answer: str
    citations: list[str]


class State(MessagesState):
    messages: Annotated[list[AnyMessage], add_messages]
    workflow_stage: str
    topic: str | None
    search_query: str | None
    documents: list[dict]
    summary: str | None
    quiz: Quiz | None
    patient_answer: str | None
    evaluation: QuizEvaluation | None
    feedback: str | None


mem_serial = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("__main__", "Quiz"),
        ("__main__", "QuizEvaluation"),
    ]
)


@tool
def search_tavily(query: str) -> dict:
    """
    Used to perform a web search for the query passed to tool
    Returns a series or related documents relevant to the query asked
    """

    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    results = tavily_client.search(
        query=query,
        max_results=5,
        search_depth="advanced",
    )

    return {"documents": results["results"]}


tool_ls = [search_tavily]
search_llm = llm.bind_tools(tool_ls, tool_choice="search_tavily")


def entry_point(state: State, config: RunnableConfig):
    """
    Sets up the system message and initialises the LLM and messages
    """
    messages = state["messages"]
    messages = [
        SystemMessage(content=GLOBAL_SYSTEM_PROMPT),
    ]

    ai_message = llm.invoke(messages)

    return {
        "messages": [ai_message],
        "workflow_stage": "requesting_topic",
    }


def ask_patient(state: State):
    topic = interrupt(
        {
            "type": "request_topic",
            "message": (
                "What health topic or medical condition would you like to learn about?"
            ),
        }
    )

    return {
        "messages": [HumanMessage(content=str(topic))],
        "topic": str(topic),
        "workflow_stage": "topic_received",
    }


def research_topic(state: State):
    """
    Invokes the search tool with a predefined query and stores to the state and appends messages
    """
    messages = [
        SystemMessage(content=SEARCH_QUERY_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Search for current medical information about: {state['topic']}"
        ),
    ]

    ai_message = search_llm.invoke(messages)

    return {
        "messages": [ai_message],
        "search_query": state["topic"],
        "workflow_stage": "researching_topic",
    }


def summarise_docs(state: State):
    """
    Retrieves the tool response content and uses the llm to summarise and saves to state
    """
    tool_messages = [
        message for message in state["messages"] if isinstance(message, ToolMessage)
    ]

    source_text = tool_messages[-1].content

    messages = [
        SystemMessage(content=GLOBAL_SYSTEM_PROMPT + "\n\n" + SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=(f"Topic:\n{state['topic']}\n\nSources:\n{source_text}")),
    ]

    ai_response = llm.invoke(messages)

    return {
        "messages": [ai_response],
        "summary": ai_response.content,
        "workflow_stage": "awaiting_quiz_readiness",
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

    quiz_model = llm.with_structured_output(Quiz)

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


def format_evaluation(eval: QuizEvaluation) -> str:
    """
    Performs some conditional formatting on the feedback to the quiz based on if it was answered correct or not.
    """

    fmt = f"That is {eval.grade}. {eval.explanation}\n\n"

    if eval.grade != "correct":
        fmt += f"The correct answer is: {eval.correct_answer}\n\n"

    fmt += f"Citations: {','.join(eval.citations)}"
    return fmt


def evaluate_answer(state: State) -> Command:
    """
    Provides the summary, quiz and answer to the LLM for a structured response.
    """

    quiz = state["quiz"]
    patient_answer = state["patient_answer"]
    summary = state["summary"]

    messages = [
        SystemMessage(content=GLOBAL_SYSTEM_PROMPT + "\n\n" + EVALUATION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Summary:\n{summary}\n\nQuiz:\n{quiz}\n\nAnswer:\n{patient_answer}"
            )
        ),
    ]

    evaluation_llm = llm.with_structured_output(QuizEvaluation)
    evaluation = evaluation_llm.invoke(messages)
    formatted_response = format_evaluation(evaluation)

    return Command(
        update={
            "feedback": formatted_response,
            "evaluation": evaluation,
            "workflow_stage": "evaluation_complete",
        },
        goto="continue_or_exit",
    )


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
                "search_query": None,
                "documents": [],
                "summary": None,
                "quiz": None,
                "patient_answer": None,
                "workflow_stage": "requesting_topic",
            },
            goto="ask_patient",
        )

    return Command(goto=END)


def reset_for_next_topic(state: State):
    return {
        "workflow_stage": "requesting_topic",
        "topic": None,
        "search_query": None,
        "documents": [],
        "summary": None,
        "quiz": None,
        "patient_answer": None,
        "evaluation": None,
    }


def search_router(state: State):
    """
    Router for calling the search tool
    """

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "summarise_docs"


workflow = StateGraph(State)

workflow.add_node("entry_point", entry_point)
workflow.add_node("ask_patient", ask_patient)
workflow.add_node("research_topic", research_topic)
workflow.add_node("summarise_docs", summarise_docs)
workflow.add_node("quiz_wait", quiz_wait)
workflow.add_node("generate_quiz", generate_quiz)
workflow.add_node("quiz_answer", quiz_answer)
workflow.add_node("evaluate_answer", evaluate_answer)
workflow.add_node("tools", ToolNode(tool_ls))
workflow.add_node("continue_or_exit", continue_or_exit)

workflow.add_edge(START, "entry_point")
workflow.add_edge("entry_point", "ask_patient")
workflow.add_edge("ask_patient", "research_topic")
workflow.add_conditional_edges(
    source="research_topic", path=search_router, path_map=["tools", "summarise_docs"]
)
workflow.add_edge("tools", "summarise_docs")
workflow.add_edge("summarise_docs", "quiz_wait")
workflow.add_edge("quiz_wait", "generate_quiz")
workflow.add_edge("generate_quiz", "quiz_answer")
workflow.add_edge("quiz_answer", "evaluate_answer")


graph = workflow.compile(checkpointer=InMemorySaver(serde=mem_serial))

config = {"configurable": {"thread_id": "patient-session-123"}}

result = graph.invoke(
    {
        "messages": [],
        "workflow_stage": "started",
        "topic": None,
        "search_query": None,
        "documents": [],
        "summary": None,
        "quiz": None,
        "patient_answer": None,
        "evaluation": None,
    },
    config=config,
)


if __name__ == "__main__":
    """
    Basic execution of the workflow within a while loop.
    """

    result = graph.invoke(
        {
            "messages": [],
            "workflow_stage": "started",
            "topic": None,
            "search_query": None,
            "documents": [],
            "summary": None,
            "quiz": None,
            "patient_answer": None,
            "evaluation": None,
            "feedback": None,
            "quiz_ready": None,
        },
        config=config,
    )

    print()
    print("@@@ Welcome to HealthBot CLI @@@")

    while True:
        interrupts = result.get("__interrupt__")

        if not interrupts:
            print()
            print("HealthBot session ended. Goodbye.")
            break

        payload = interrupts[-1].value
        interrupt_type = payload["type"]

        print()

        if interrupt_type == "request_topic":
            print(payload["message"])
            user_input = input("\nTopic: ")

        elif interrupt_type == "confirmation":
            print("=== SUMMARY ===")
            print(payload["message1"])
            print()
            print(payload["message2"])
            user_input = input("\nReady? [y/n]: ").strip().lower() == "y"

        elif interrupt_type == "quiz_answer":
            print("QUIZ:", payload["message"])
            user_input = input("\nAnswer: ")

        elif interrupt_type == "continue_or_exit":
            print("=== FEEDBACK ===")
            print(payload["feedback"])
            print("-----------------")
            print(payload["message"])
            user_input = input("Continue? [y/n]: ").strip().lower()

        else:
            raise ValueError(f"Unknown interrupt type: {interrupt_type}")

        result = graph.invoke(
            Command(resume=user_input),
            config=config,
        )
