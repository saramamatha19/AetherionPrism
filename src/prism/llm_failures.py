"""Where does the LLM disagree with our labels? Studied on the PRACTICE set (train), never eval (Phase 5).

Run:  uv run python -m prism.llm_failures                       (model + prompt from YAML)
      uv run python -m prism.llm_failures <model> <prompt version> [timeout seconds]   (try others)
      Records the practice set first if needed (~196 questions).
"""

import sys

from prism.data import SOURCES


def failure_report(questions, llm_answers) -> dict:
    """Where the LLM disagrees with our labels: by source (added / missed) and the top confusions."""
    added_wrongly = {}
    missed = {}
    for source in SOURCES:
        added_wrongly[source] = 0
        missed[source] = 0
    confusions = {}
    failed_examples = []

    for q in questions:
        llm = llm_answers[q.id]
        if llm == q.labels:
            continue  # exactly right

        for source in llm - q.labels:  # the LLM said it, our label didn't
            added_wrongly[source] += 1
        for source in q.labels - llm:  # our label has it, the LLM didn't say it
            missed[source] += 1

        key = (tuple(sorted(q.labels)), tuple(sorted(llm)))  # e.g. (("kb",), ("confluence", "kb"))
        if key not in confusions:
            confusions[key] = 0
        confusions[key] += 1

        failed_examples.append((q.question_type, q.text, sorted(q.labels), sorted(llm)))

    return {
        "added_wrongly": added_wrongly,
        "missed": missed,
        "confusions": confusions,
        "failed_examples": failed_examples,
    }


if __name__ == "__main__":
    from prism.config import load_settings
    from prism.data import load_questions, prompt_practice_set
    from prism.evaluate import evaluate, print_table
    from prism.llm_answer_eval import load_llm_answers, record_llm_answers

    settings = load_settings()
    # Optional: try another model / prompt version without editing the YAML (for experiments only)
    if len(sys.argv) >= 3:
        override = {"llm_model": sys.argv[1], "llm_prompt_version": sys.argv[2]}
        if len(sys.argv) >= 4:
            override["llm_timeout_seconds"] = float(sys.argv[3])  # reasoning models think longer
        settings = settings.model_copy(update=override)

    practice = prompt_practice_set(
        load_questions("data/train.csv"), settings.prompt_practice_per_type, settings.seed
    )
    print(
        f"Practice set: {len(practice)} train questions │ {settings.llm_model} │ prompt {settings.llm_prompt_version}"
    )
    record_llm_answers(practice, settings, question_set="practice")
    answers = load_llm_answers(practice, settings, question_set="practice")

    # 1. By question type (the same marking as everywhere else)
    answers_by_text = {}
    for q in practice:
        answers_by_text[q.text] = answers[q.id]

    def llm_student(text):
        return answers_by_text[text]

    result = evaluate(llm_student, practice)
    print_table("By question type", {**result["by_type"], "ALL": result["all"]})

    # 2. By source: added wrongly / missed
    report = failure_report(practice, answers)
    print(f"\n{'By source':<14}{'added wrongly':>15}{'missed':>9}")
    for source in SOURCES:
        print(f"{source:<14}{report['added_wrongly'][source]:>15}{report['missed'][source]:>9}")

    # 3. Top confusions
    print("\nTop confusions (our label → LLM said):")
    ordered = sorted(report["confusions"].items(), key=lambda item: -item[1])
    for (label, llm), count in ordered[:12]:
        print(f"  {count:>3} ×  {list(label) or '(none)'} → {list(llm) or '(none)'}")

    # Cost and speed on the practice set (real token counts)
    from prism.compare import percentile
    from prism.cost import llm_call_cost
    from prism.llm_answer_eval import answers_file, read_answers

    saved = read_answers(answers_file(settings, "practice"))
    total = 0.0
    latencies = []
    completion = []
    for q in practice:
        r = saved[q.id]
        total += llm_call_cost(
            r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"], settings
        )
        latencies.append(r["latency_ms"])
        completion.append(r["completion_tokens"])
    print(
        f"\nCost ${total:.3f} (per question ${total / len(practice):.5f}) │ "
        f"speed p50 {percentile(latencies, 50):.0f} ms, p95 {percentile(latencies, 95):.0f} ms │ "
        f"output tokens p50 {percentile(completion, 50):.0f}"
    )

    # 4. Failed examples to read
    print("\n20 failed examples:")
    for question_type, text, label, llm in report["failed_examples"][:20]:
        print(
            f"  [{question_type}] {text[:90]}\n      label {label or '(none)'} │ LLM {llm or '(none)'}"
        )
