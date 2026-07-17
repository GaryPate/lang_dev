from langgraph.types import Command

from healthbot.graph import build_graph, initial_state
from healthbot.observability import flush_langfuse


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "patient-session-123"}}
    result = graph.invoke(initial_state(), config=config)

    print()
    print("@@@ Welcome to HealthBot CLI @@@")

    while True:
        interrupts = result.get("__interrupt__")
        if not interrupts:
            print()
            print("HealthBot session ended. Goodbye.")
            flush_langfuse()
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

        result = graph.invoke(Command(resume=user_input), config=config)


if __name__ == "__main__":
    main()
