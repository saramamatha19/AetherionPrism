"""Report on the LLM's recorded eval answers: how good (vs our labels) and what it costs (Phase 5).

Run:  uv run python -m prism.llm_report      (free: it only reads the recording)
"""

from prism.compare import percentile
from prism.cost import llm_call_cost, model_prices
from prism.evaluate import evaluate, print_report
from prism.llm_answer_eval import answers_file, load_llm_answers, read_answers


def make_llm_student(questions, settings):
    """The recorded LLM answers as a "student": question text → sources (replayed, free)."""
    answers_by_id = load_llm_answers(questions, settings)  # error if anything is missing
    answers_by_text = {}
    for q in questions:
        answers_by_text[q.text] = answers_by_id[q.id]

    def llm_student(text):
        return answers_by_text[text]

    return llm_student


def llm_cost_and_speed(questions, settings) -> dict:
    """Total cost and speed of the recorded LLM answers, from their real token counts."""
    saved = read_answers(answers_file(settings, "eval"))
    total_cost = 0.0
    latencies = []
    for q in questions:
        record = saved[q.id]
        total_cost += llm_call_cost(
            record["prompt_tokens"], record["cached_tokens"], record["completion_tokens"], settings
        )
        latencies.append(record["latency_ms"])
    per_question = total_cost / len(questions)
    return {
        "total": total_cost,
        "per_question": per_question,
        "per_million_questions": per_question * 1_000_000,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
    }


if __name__ == "__main__":
    from prism.config import load_settings
    from prism.data import load_questions

    settings = load_settings()
    questions = load_questions("data/eval.csv")

    result = evaluate(make_llm_student(questions, settings), questions)
    print_report(f"LLM ({settings.llm_model})", result)

    money = llm_cost_and_speed(questions, settings)
    print(f"\nLLM cost (prices: {model_prices(settings).source}):")
    print(
        f"  total ${money['total']:.2f} │ per question ${money['per_question']:.5f} "
        f"│ per 1 million questions ${money['per_million_questions']:,.0f}"
    )
    print(f"  speed: p50 {money['p50_ms']:.0f} ms │ p95 {money['p95_ms']:.0f} ms")
