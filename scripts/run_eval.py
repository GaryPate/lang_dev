from healthbot.evaluation import (
    aggregate_metrics,
    deterministic_checks,
    load_eval_set,
    run_eval_case,
)


def main():
    cases = load_eval_set("eval/healthbot_eval_set.jsonl")
    evaluated = []

    for case in cases:
        result = run_eval_case(case["question"])
        checks = deterministic_checks(case, result)
        row = {
            "question": case["question"],
            "result": result,
            "checks": checks,
        }
        evaluated.append(row)
        print(case["question"])
        print(checks)
        print("---")

    print(aggregate_metrics(evaluated))


if __name__ == "__main__":
    main()
