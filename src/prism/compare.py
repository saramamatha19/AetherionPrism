"""Compare saved models side by side on eval: accuracy, coverage, hard cases, speed, size."""

import sys
import time
from pathlib import Path

from prism.confidence import confidence
from prism.data import load_questions
from prism.evaluate import evaluate
from prism.gate import coverage_at
from prism.logreg import scores
from prism.pipeline import confidences_and_correct, make_student
from prism.registry import MODELS_DIR, load_model
from prism.thresholds import apply_thresholds

RESULTS_FILE = Path("results/model_comparison.md")


def percentile(values: list[float], p: int) -> float:
    """The value that p% of the list is below (p=50 → typical, p=95 → the slow cases)."""
    ordered = sorted(values)
    index = int(len(ordered) * p / 100)
    if index >= len(ordered):
        index = len(ordered) - 1
    return ordered[index]


def speed_ms(model, thresholds, questions) -> tuple[float, float]:
    """Time each question on its own (model already loaded) → p50 and p95 in milliseconds."""
    scores(
        model, "warm-up"
    )  # the first call loads things; not timed (plan: warm-up never counts)
    times = []
    for q in questions:
        start = time.perf_counter()
        s = scores(model, q.text)
        confidence(s, thresholds)
        apply_thresholds(s, thresholds)
        times.append((time.perf_counter() - start) * 1000)
    return percentile(times, 50), percentile(times, 95)


def gate_on_part(confidences, correct, questions, part, gate) -> dict:
    """The gate check (coverage + right on kept) for one part only, e.g. "hard cases"."""
    part_confidences = []
    part_correct = []
    for i in range(len(questions)):
        if questions[i].part == part:
            part_confidences.append(confidences[i])
            part_correct.append(correct[i])
    return coverage_at(part_confidences, part_correct, gate)


def compare_one(version: str, questions) -> dict:
    """All the numbers for one saved model."""
    model, thresholds, gate = load_model(version)
    result = evaluate(make_student(model, thresholds), questions)
    confidences, correct = confidences_and_correct(model, thresholds, questions)
    overall = coverage_at(confidences, correct, gate)
    real_world = gate_on_part(confidences, correct, questions, "real-world", gate)
    company = gate_on_part(confidences, correct, questions, "company topics", gate)
    hard = gate_on_part(confidences, correct, questions, "hard cases", gate)

    p50, p95 = speed_ms(model, thresholds, questions)
    size_mb = (MODELS_DIR / version / "model.joblib").stat().st_size / 1_000_000

    return {
        "version": version,
        # Table 1: accuracy, before the gate
        "all_exact": result["all"]["exact_match"],
        "real_world_exact": result["by_part"]["real-world"]["exact_match"],
        "company_exact": result["by_part"]["company topics"]["exact_match"],
        "hard_exact": result["by_part"]["hard cases"]["exact_match"],
        "all_needed": result["all"]["found_all_needed"],
        # Table 2: after the gate
        "gate": gate,
        "coverage": overall["coverage"],
        "kept_accuracy": overall["accuracy"],
        "real_world_coverage": real_world["coverage"],
        "real_world_kept_accuracy": real_world["accuracy"],
        "company_coverage": company["coverage"],
        "company_kept_accuracy": company["accuracy"],
        "hard_coverage": hard["coverage"],
        "hard_kept_accuracy": hard["accuracy"],
        # Table 3: speed and size
        "p50_ms": p50,
        "p95_ms": p95,
        "size_mb": size_mb,
    }


if __name__ == "__main__":
    versions = sys.argv[1:]
    if not versions:
        raise SystemExit(
            "Usage: uv run python -m prism.compare <version> <version> ..."
        )

    questions = load_questions("data/eval.csv")
    rows = []
    for version in versions:
        rows.append(compare_one(version, questions))

    table1 = [
        "### Table 1: Accuracy, before the gate",
        "",
        "| model | all exact | real-world exact | company exact | hard exact | all needed |",
        "|---|---|---|---|---|---|",
    ]
    table2 = [
        "### Table 2: After the gate",
        "",
        "| model | gate | coverage | right on kept | real-world coverage "
        "| real-world right on kept | company coverage | company right on kept "
        "| hard-case coverage | hard-case kept |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    table3 = [
        "### Table 3: Speed and size",
        "",
        "| model | p50 ms | p95 ms | size MB |",
        "|---|---|---|---|",
    ]
    table4 = [
        "### Table 4: Exact accuracy on all 1,200 questions, any category",
        "",
        "| model | before the gate (all exact) | after the gate (right on kept) |",
        "|---|---|---|",
    ]
    table5 = [
        "### Table 5: Out of 100 questions (without the gate vs with the gate)",
        "",
        "| model | questions | without gate: right | without gate: wrong "
        "| with gate: model right | with gate: model wrong | with gate: sent to LLM |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        table1.append(
            f"| {r['version']} | {r['all_exact']:.1%} | {r['real_world_exact']:.1%} "
            f"| {r['company_exact']:.1%} | {r['hard_exact']:.1%} | {r['all_needed']:.1%} |"
        )
        table2.append(
            f"| {r['version']} | {r['gate']} | {r['coverage']:.1%} | {r['kept_accuracy']:.1%} "
            f"| {r['real_world_coverage']:.1%} | {r['real_world_kept_accuracy']:.1%} "
            f"| {r['company_coverage']:.1%} | {r['company_kept_accuracy']:.1%} "
            f"| {r['hard_coverage']:.1%} | {r['hard_kept_accuracy']:.1%} |"
        )
        table3.append(
            f"| {r['version']} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['size_mb']:.2f} |"
        )
        table4.append(
            f"| {r['version']} | {r['all_exact']:.1%} | {r['kept_accuracy']:.1%} |"
        )
        parts = [
            ("all", r["all_exact"], r["coverage"], r["kept_accuracy"]),
            (
                "real-world",
                r["real_world_exact"],
                r["real_world_coverage"],
                r["real_world_kept_accuracy"],
            ),
            (
                "company",
                r["company_exact"],
                r["company_coverage"],
                r["company_kept_accuracy"],
            ),
            (
                "hard cases",
                r["hard_exact"],
                r["hard_coverage"],
                r["hard_kept_accuracy"],
            ),
        ]
        for name, exact, coverage, kept_accuracy in parts:
            model_right = coverage * kept_accuracy * 100
            model_wrong = coverage * (1 - kept_accuracy) * 100
            to_llm = (1 - coverage) * 100
            table5.append(
                f"| {r['version']} | {name} | {exact * 100:.0f} | {(1 - exact) * 100:.0f} "
                f"| {model_right:.0f} | {model_wrong:.0f} | {to_llm:.0f} |"
            )

    tables = ""
    for table in [table1, table2, table3, table4, table5]:
        tables += "\n".join(table) + "\n\n"
    print(tables)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text("# Model comparison (eval, 1,200 questions)\n\n" + tables)
    print(f"\nSaved to {RESULTS_FILE}")
