"""The full system on eval: the model answers when sure, the LLM otherwise (Phase 5.6).

Free: the LLM's answers are replayed from the recording (never asked again).
Run:  uv run python -m prism.system_eval tfidf_ftv2_bge_logreg_v2          (the full system)
      uv run python -m prism.system_eval tfidf_ftv2_bge_logreg_v2 curve    (cost–accuracy curve)
"""
# creates cost_accuracy_curve.md in results/

from prism.confidence import confidence
from prism.cost import llm_call_cost
from prism.llm_answer_eval import answers_file, load_llm_answers, read_answers
from prism.logreg import scores
from prism.registry import load_card, load_model
from prism.ruleNegation import has_negation
from prism.thresholds import apply_thresholds


def route_one(
    model_confidence: float, gate: float, question_has_negation: bool, negation_rule_on: bool
) -> str:
    """Who answers ONE question. The same order the real router (Phase 6) will use:
    1. the negation rule first (known weak spot → always the LLM)
    2. then the gate (sure → model, unsure → LLM)
    """
    if negation_rule_on and question_has_negation:
        return "llm (negation rule)"
    if model_confidence >= gate:
        return "model"
    return "llm (unsure)"


def full_system_numbers(
    questions, answers, confidences, llm_answers, llm_costs, gate, rule_on
) -> dict:
    """The full system with ANY small model (Laya or model B): it answers when sure, the LLM otherwise.
    Plain inputs keyed by question id, so model B and Laya are scored by exactly the same code."""
    right = 0
    needed = 0
    per_part = {}
    small_model_count = 0
    hard_kept = 0
    hard_kept_wrong = 0
    hard_count = 0
    total_cost = 0.0
    for q in questions:
        decided_by = route_one(confidences[q.id], gate, has_negation(q.text), rule_on)
        if decided_by == "model":
            final = answers[q.id]
            small_model_count += 1
        else:
            final = llm_answers[q.id]
            total_cost += llm_costs[q.id]
        is_right = final == q.labels
        right += is_right
        needed += q.labels <= final
        if q.part not in per_part:
            per_part[q.part] = [0, 0]
        per_part[q.part][0] += is_right
        per_part[q.part][1] += 1
        if q.part == "hard cases":
            hard_count += 1
            if decided_by == "model":
                hard_kept += 1
                hard_kept_wrong += not is_right

    n = len(questions)
    return {
        "all_exact": right / n,
        "real_world": per_part["real-world"][0] / per_part["real-world"][1],
        "company": per_part["company topics"][0] / per_part["company topics"][1],
        "hard": per_part["hard cases"][0] / per_part["hard cases"][1],
        "all_needed": needed / n,
        "small_model_share": small_model_count / n,
        "hard_confident_mistakes_per_100": hard_kept_wrong / hard_count * 100,
        "cost_per_million": total_cost / n * 1_000_000,
    }


def system_decisions(version: str, questions, settings) -> list[dict]:
    """For each question: who decides, what each one said, the final answer, and its cost."""
    model, thresholds, gate = load_model(version)
    negation_rule_on = load_card(version)["settings"].get("negation_to_llm", False)
    llm_answers = load_llm_answers(questions, settings)  # error if any are missing
    llm_records = read_answers(answers_file(settings, "eval"))  # for the real token counts

    decisions = []
    for q in questions:
        model_scores = scores(model, q.text)
        model_answer = apply_thresholds(model_scores, thresholds)
        model_confidence, _ = confidence(model_scores, thresholds)
        decided_by = route_one(model_confidence, gate, has_negation(q.text), negation_rule_on)

        if decided_by == "model":
            final_answer = model_answer
            cost = 0.0  # the model is free
        else:
            final_answer = llm_answers[q.id]
            record = llm_records[q.id]
            cost = llm_call_cost(
                record["prompt_tokens"],
                record["cached_tokens"],
                record["completion_tokens"],
                settings,
            )

        decisions.append(
            {
                "question": q,
                "decided_by": decided_by,
                "model_answer": model_answer,  # always kept (plan B8), even when the LLM decided
                "model_confidence": model_confidence,
                "llm_answer": llm_answers[q.id],
                "final_answer": final_answer,
                "cost": cost,
            }
        )
    return decisions


def summarise(decisions, all_llm_cost: float) -> dict:
    """Accuracy, who decided, model-vs-LLM on the LLM's questions, and cost vs LLM-for-everything."""
    n = len(decisions)
    exact = 0
    needed = 0
    by_part = {}
    who = {"model": 0, "llm (unsure)": 0, "llm (negation rule)": 0}
    llm_questions = 0
    model_right_on_llm_questions = 0
    llm_right_on_llm_questions = 0
    total_cost = 0.0

    for d in decisions:
        q = d["question"]
        right = d["final_answer"] == q.labels
        exact += right
        needed += q.labels <= d["final_answer"]
        if q.part not in by_part:
            by_part[q.part] = [0, 0]
        by_part[q.part][0] += right
        by_part[q.part][1] += 1
        who[d["decided_by"]] += 1
        total_cost += d["cost"]
        if d["decided_by"] != "model":
            llm_questions += 1
            model_right_on_llm_questions += d["model_answer"] == q.labels
            llm_right_on_llm_questions += d["llm_answer"] == q.labels

    part_exact = {}
    for part, (right_count, count) in by_part.items():
        part_exact[part] = right_count / count

    return {
        "all_exact": exact / n,
        "all_needed": needed / n,
        "part_exact": part_exact,
        "who": who,
        "llm_questions": llm_questions,
        "model_right_on_llm_questions": model_right_on_llm_questions / llm_questions,
        "llm_right_on_llm_questions": llm_right_on_llm_questions / llm_questions,
        "cost_per_million": total_cost / n * 1_000_000,
        "llm_everything_cost_per_million": all_llm_cost / n * 1_000_000,
    }


def curve_points(
    questions, model_answers, model_confidences, llm_answers, llm_costs, gates, negation_rule_on
) -> list[dict]:
    """For each gate: who answers, accuracy and cost. Plain inputs (no models, no files), so it
    can be tested with tiny made-up data. All dicts are keyed by question id."""
    curve = []
    for gate in gates:
        exact = 0
        real_world_right = 0
        real_world_count = 0
        model_count = 0
        total_cost = 0.0
        for q in questions:
            decided_by = route_one(
                model_confidences[q.id], gate, has_negation(q.text), negation_rule_on
            )
            if decided_by == "model":
                final_answer = model_answers[q.id]
                model_count += 1
            else:
                final_answer = llm_answers[q.id]
                total_cost += llm_costs[q.id]
            exact += final_answer == q.labels
            if q.part == "real-world":
                real_world_count += 1
                real_world_right += final_answer == q.labels

        real_world_exact = None
        if real_world_count > 0:
            real_world_exact = real_world_right / real_world_count

        curve.append(
            {
                "gate": gate,
                "model_share": model_count / len(questions),
                "all_exact": exact / len(questions),
                "real_world_exact": real_world_exact,
                "cost_per_million": total_cost / len(questions) * 1_000_000,
            }
        )
    return curve


def cost_accuracy_curve(version: str, questions, settings, gates: list[float]) -> list[dict]:
    """The full system at many different gates: accuracy vs cost (free: uses the recording).

    Today's gate is chosen by the 80% rule; this shows what other gates would give, so the
    trade-off "pay more → more accurate?" can be seen with real numbers.
    """
    model, thresholds, _real_gate = load_model(version)
    negation_rule_on = load_card(version)["settings"].get("negation_to_llm", False)
    llm_answers = load_llm_answers(questions, settings)
    llm_records = read_answers(answers_file(settings, "eval"))

    # Score every question once: the model's answer doesn't depend on the gate
    model_answers = {}
    model_confidences = {}
    llm_costs = {}
    for q in questions:
        model_scores = scores(model, q.text)
        model_answers[q.id] = apply_thresholds(model_scores, thresholds)
        model_confidences[q.id], _ = confidence(model_scores, thresholds)
        r = llm_records[q.id]
        llm_costs[q.id] = llm_call_cost(
            r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"], settings
        )

    return curve_points(
        questions, model_answers, model_confidences, llm_answers, llm_costs, gates, negation_rule_on
    )


if __name__ == "__main__":
    import sys

    from prism.config import load_settings
    from prism.data import load_questions
    from prism.evaluate import evaluate
    from prism.llm_report import llm_cost_and_speed, make_llm_student
    from prism.pipeline import make_student

    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: uv run python -m prism.system_eval <model version> [curve]")
    version = sys.argv[1]

    settings = load_settings()
    questions = load_questions("data/eval.csv")

    if len(sys.argv) == 3 and sys.argv[2] == "curve":
        from pathlib import Path

        _, _, real_gate = load_model(version)
        gates = [0.0, 0.05, 0.10, real_gate, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 0.95]
        gates = sorted(set(gates))
        lines = [
            f"# Cost–accuracy curve: {version} + {settings.llm_model} prompt {settings.llm_prompt_version}",
            "",
            "| gate | model answers | all exact | real-world exact | cost per 1M questions |",
            "|---|---|---|---|---|",
        ]
        for point in cost_accuracy_curve(version, questions, settings, gates):
            mark = ""
            if point["gate"] == real_gate:
                mark = " ← today (80% rule)"
            lines.append(
                f"| {point['gate']:.2f}{mark} | {point['model_share']:.0%} | {point['all_exact']:.1%} "
                f"| {point['real_world_exact']:.1%} | ${point['cost_per_million']:,.0f} |"
            )
        table = "\n".join(lines)
        print(table)
        Path("results").mkdir(exist_ok=True)
        Path("results/cost_accuracy_curve.md").write_text(table + "\n")
        print("\nSaved to results/cost_accuracy_curve.md")
        raise SystemExit(0)

    # The two "pure" options to compare against
    model, thresholds, _ = load_model(version)
    model_only = evaluate(make_student(model, thresholds), questions)
    llm_only = evaluate(make_llm_student(questions, settings), questions)
    llm_cost = llm_cost_and_speed(questions, settings)["total"]

    s = summarise(system_decisions(version, questions, settings), llm_cost)

    print(f"\nModel {version} + LLM {settings.llm_model} prompt {settings.llm_prompt_version}")
    print("\nTable 1: accuracy (eval, 1,200)")
    print(f"{'':<22}{'all exact':>10}{'real-world':>12}{'company':>9}{'hard':>7}{'all needed':>12}")
    for name, r in [("model for everything", model_only), ("LLM for everything", llm_only)]:
        bp = r["by_part"]
        print(
            f"{name:<22}{r['all']['exact_match']:>10.1%}{bp['real-world']['exact_match']:>12.1%}"
            f"{bp['company topics']['exact_match']:>9.1%}{bp['hard cases']['exact_match']:>7.1%}"
            f"{r['all']['found_all_needed']:>12.1%}"
        )
    pe = s["part_exact"]
    print(
        f"{'FULL SYSTEM':<22}{s['all_exact']:>10.1%}{pe['real-world']:>12.1%}"
        f"{pe['company topics']:>9.1%}{pe['hard cases']:>7.1%}{s['all_needed']:>12.1%}"
    )

    print("\nTable 2: who decided")
    for name, count in s["who"].items():
        print(f"  {name:<22}{count:>5}  ({count / len(questions):.0%})")
    print(
        f"  On the {s['llm_questions']} questions the LLM answered: "
        f"model would be right {s['model_right_on_llm_questions']:.0%} │ "
        f"LLM right {s['llm_right_on_llm_questions']:.0%}"
    )

    saved = 1 - s["cost_per_million"] / s["llm_everything_cost_per_million"]
    print("\nTable 3: cost per 1 million questions")
    print(f"  LLM for everything   ${s['llm_everything_cost_per_million']:,.0f}")
    print(f"  FULL SYSTEM          ${s['cost_per_million']:,.0f}   → {saved:.0%} cheaper")
