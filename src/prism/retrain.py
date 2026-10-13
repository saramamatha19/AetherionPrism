"""The learning loop (Phase 8): collect reviewed questions → train a new version → compare → promote.

Run:  uv run python -m prism.retrain collect <shadow run folder>
      uv run python -m prism.retrain train          (train a challenger, then compare it with the champion)
      uv run python -m prism.retrain promote <version> "<what it is>" "<why>"
      uv run python -m prism.retrain rollback "<why>"
      uv run python -m prism.retrain show
"""

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from prism.data import load_questions
from prism.registry import MODELS_DIR, promote, read_champion, rollback
from prism.shadowReport import as_set, load_run, read_review_file

NEW_LABELS_DIR = Path("data/new_labels")


def normalise(text: str) -> str:
    """For comparing questions: lowercase, single spaces. "Is the  bug fixed?" = "is the bug fixed?"."""
    return " ".join(text.lower().split())


def known_questions(new_labels_dir: Path) -> set:
    """Every question we ALREADY train on: train.csv + all earlier batches (to avoid duplicates)."""
    known = set()
    for q in load_questions("data/train.csv"):
        known.add(normalise(q.text))
    for path in sorted(new_labels_dir.glob("batch_*.csv")):
        for q in load_questions(path):
            known.add(normalise(q.text))
    return known


def collect(log_dir: Path, reviewer: str, new_labels_dir: Path = NEW_LABELS_DIR) -> dict:
    """A reviewed shadow run → data/new_labels/batch_<date>.csv. Only checked labels; never eval."""
    log_dir = Path(log_dir)
    decisions = load_run(log_dir)  # shadow-mode lines only

    correct_answers = {}  # question → your answer, from review.csv
    review_path = log_dir / "review.csv"
    if review_path.exists():
        for row in read_review_file(review_path):  # error if any row is still empty
            correct_answers[row["question"]] = as_set(row["correct_answer"])

    # Safety 1: NEVER the exam. Any eval question → stop everything (something went wrong upstream).
    eval_texts = set()
    for q in load_questions("data/eval.csv"):
        eval_texts.add(normalise(q.text))
    in_eval = []
    for d in decisions:
        if normalise(d["question"]) in eval_texts:
            in_eval.append(d["question"])
    if in_eval:
        raise ValueError(
            f"{len(in_eval)} of these questions are EVAL questions (e.g. {in_eval[0]!r}): "
            "never train on the exam. Is this the rehearsal folder?"
        )

    already_known = known_questions(new_labels_dir)  # Safety 2: no duplicates
    rows = []
    counts = {
        "from_review": 0,
        "from_agreement": 0,
        "skipped_unreviewed": 0,
        "skipped_duplicate": 0,
    }
    for d in decisions:
        text = d["question"]
        if normalise(text) in already_known:
            counts["skipped_duplicate"] += 1
            continue

        llm_answered = d["llm_answer"] is not None
        if text in correct_answers:  # you checked it: the best label
            labels = correct_answers[text]
            counts["from_review"] += 1
        elif llm_answered and set(d["model_answer"]) == set(d["llm_answer"]):
            labels = frozenset(d["model_answer"])  # model and LLM agree (~93% right)
            counts["from_agreement"] += 1
        else:  # they disagree and nobody checked → we don't know the right answer
            counts["skipped_unreviewed"] += 1
            continue

        already_known.add(normalise(text))
        rows.append((text, labels))

    # Write the batch (same columns as train.csv, so the same code can read it)
    today = datetime.now().strftime("%Y-%m-%d")
    path = new_labels_dir / f"batch_{today}.csv"
    if path.exists():
        raise ValueError(f"{path} already exists: one batch per day, never overwritten")
    new_labels_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "question", "labels", "question_type", "origin", "split"])
        number = 0
        for text, labels in rows:
            number += 1
            writer.writerow(
                [
                    f"n{today.replace('-', '')}_{number:04d}",  # e.g. n20261120_0001
                    text,
                    "|".join(sorted(labels)),
                    "(unknown)",  # real questions have no question type
                    f"shadow run {log_dir.name}, reviewed by {reviewer}",
                    "train",
                ]
            )
    counts["kept"] = len(rows)
    counts["file"] = str(path)
    return counts


def load_new_labels(new_labels_dir: Path = NEW_LABELS_DIR) -> tuple[list, list]:
    """All reviewed batches as Question objects (the same format as train.csv) + their file names."""
    questions = []
    batch_names = []
    for path in sorted(new_labels_dir.glob("batch_*.csv")):
        questions += load_questions(path)
        batch_names.append(path.name)
    return questions, batch_names


def train_challenger(
    settings, new_labels_dir: Path = NEW_LABELS_DIR, models_dir: Path = MODELS_DIR
) -> str:
    """The champion's recipe + the fit slice + ALL reviewed batches → a new saved version.

    The only difference from the champion is the new data (same recipe, same threshold and
    gate slices), so the comparison is fair.
    """
    from prism.data import load_questions as load
    from prism.evaluate import evaluate
    from prism.gate import coverage_at
    from prism.pipeline import (
        SPLIT_FIX_RECIPES,
        build_features,
        confidences_and_correct,
        make_student,
        routing_confidences,
        train_small_model,
    )
    from prism.registry import save_model

    new_questions, batch_names = load_new_labels(new_labels_dir)
    if len(new_questions) < settings.retrain_min_new_rows:
        raise SystemExit(
            f"Only {len(new_questions)} new reviewed rows (need {settings.retrain_min_new_rows}, "
            "retrain_min_new_rows in the YAML): not enough new data to learn from yet."
        )

    champion_card = json.loads((MODELS_DIR / read_champion()["version"] / "card.json").read_text())
    recipe = champion_card["recipe"]  # e.g. "tfidf_ftv2_bge_logreg"
    feature_recipe = recipe.removesuffix("_logreg")  # e.g. "tfidf_ftv2_bge"

    print(f"Training a challenger: {recipe} + {len(new_questions)} new rows from {batch_names}")
    model, thresholds, gate, gate_slice_result = train_small_model(
        settings,
        features=build_features(feature_recipe),
        judges_on_judges_half=(feature_recipe in SPLIT_FIX_RECIPES),
        extra_fit_questions=new_questions,
    )

    eval_questions = load("data/eval.csv")
    eval_result = evaluate(make_student(model, thresholds), eval_questions)
    confidences, correct = confidences_and_correct(model, thresholds, eval_questions)
    routing = routing_confidences(confidences, eval_questions, settings.negation_to_llm)
    eval_gate_result = coverage_at(routing, correct, gate)

    return save_model(
        model,
        thresholds,
        gate,
        recipe,
        settings,
        eval_result,
        gate_slice_result,
        eval_gate_result,
        models_dir=models_dir,
        extra_card={"learned_from_batches": batch_names, "new_rows": len(new_questions)},
    )


def full_system_on_eval(version: str, settings, models_dir: Path) -> dict:
    """One saved model inside the full system on eval (recorded LLM answers: free)."""
    from prism.confidence import confidence
    from prism.cost import llm_call_cost
    from prism.llm_answer_eval import answers_file, load_llm_answers, read_answers
    from prism.logreg import scores
    from prism.registry import load_card, load_model
    from prism.system_eval import full_system_numbers
    from prism.thresholds import apply_thresholds

    eval_questions = load_questions("data/eval.csv")
    model, thresholds, gate = load_model(version, models_dir=models_dir)
    rule_on = load_card(version, models_dir=models_dir)["settings"].get("negation_to_llm", False)

    answers = {}
    confidences = {}
    for q in eval_questions:
        model_scores = scores(model, q.text)
        answers[q.id] = apply_thresholds(model_scores, thresholds)
        confidences[q.id], _ = confidence(model_scores, thresholds)

    llm_answers = load_llm_answers(eval_questions, settings)
    llm_records = read_answers(answers_file(settings, "eval"))
    llm_costs = {}
    for q in eval_questions:
        r = llm_records[q.id]
        llm_costs[q.id] = llm_call_cost(
            r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"], settings
        )
    return full_system_numbers(
        eval_questions, answers, confidences, llm_answers, llm_costs, gate, rule_on
    )


def compare_with_champion(
    challenger: str, settings, challenger_dir: Path = MODELS_DIR, run_tests: bool = True
) -> dict:
    """Champion vs challenger, scored by the SAME code → the promotion rule."""
    champion = read_champion()["version"]
    before = full_system_on_eval(champion, settings, MODELS_DIR)
    after = full_system_on_eval(challenger, settings, challenger_dir)

    checks = {
        "full-system all exact ≥ the champion's": after["all_exact"] >= before["all_exact"],
        "full-system real-world: no drop": after["real_world"] >= before["real_world"],
        "hard-case confident mistakes: no increase": after["hard_confident_mistakes_per_100"]
        <= before["hard_confident_mistakes_per_100"],
    }
    if run_tests:
        tests = subprocess.run(["uv", "run", "pytest", "-q"], capture_output=True, text=True)
        checks["all tests pass"] = tests.returncode == 0

    can_promote = True
    for passed in checks.values():
        if not passed:
            can_promote = False
    return {
        "champion": champion,
        "challenger": challenger,
        "before": before,
        "after": after,
        "checks": checks,
        "can_promote": can_promote,
    }


def print_comparison(result: dict) -> None:
    """The comparison table, the checks, and what to do next."""
    print(f"\n{'(eval, full system)':<36}{'champion':>12}{'challenger':>12}")
    for key, label in [
        ("all_exact", "all exact"),
        ("real_world", "real-world exact"),
        ("hard", "hard-case exact"),
        ("all_needed", "all needed"),
        ("small_model_share", "model answers alone"),
    ]:
        print(f"  {label:<34}{result['before'][key]:>12.1%}{result['after'][key]:>12.1%}")
    print(
        f"  {'hard-case confident mistakes /100':<34}"
        f"{result['before']['hard_confident_mistakes_per_100']:>12.0f}"
        f"{result['after']['hard_confident_mistakes_per_100']:>12.0f}"
    )
    print(
        f"  {'cost per 1M questions':<34}"
        f"{'$' + format(result['before']['cost_per_million'], ',.0f'):>12}"
        f"{'$' + format(result['after']['cost_per_million'], ',.0f'):>12}"
    )
    print()
    for check, passed in result["checks"].items():
        mark = "✅" if passed else "❌"
        print(f"  {mark} {check}")
    if result["can_promote"]:
        print(f"\nCAN be promoted. If you agree, run:")
        print(
            f"  uv run python -m prism.retrain promote {result['challenger']} "
            '"<what it is>" "<why, in plain words>"'
        )
    else:
        print(
            f"\nNOT better: keep the champion ({result['champion']}). The challenger stays saved."
        )


if __name__ == "__main__":
    command = ""
    if len(sys.argv) >= 2:
        command = sys.argv[1]

    if command == "train" and len(sys.argv) == 2:
        from prism.config import load_settings

        settings = load_settings()
        challenger = train_challenger(settings)
        print_comparison(compare_with_champion(challenger, settings))
    elif command == "collect" and len(sys.argv) == 3:
        summary = collect(Path(sys.argv[2]), reviewer=input("Your name (the reviewer): ").strip())
        print(json.dumps(summary, indent=2))
    elif command == "promote" and len(sys.argv) == 5:
        version = sys.argv[2]
        card = json.loads((MODELS_DIR / version / "card.json").read_text())
        champion = promote(
            version,
            by=input("Your name: ").strip(),
            what_it_is=sys.argv[3],
            why=sys.argv[4],
            evidence=card["eval"],  # the version's own saved eval numbers
            details=f"models/{version}/card.json",
        )
        print(json.dumps(champion, indent=2))
    elif command == "rollback" and len(sys.argv) == 3:
        champion = rollback(by=input("Your name: ").strip(), why=sys.argv[2])
        print(json.dumps(champion, indent=2))
    elif command == "show":
        print(json.dumps(read_champion(), indent=2))
    else:
        raise SystemExit(
            "Usage: uv run python -m prism.retrain collect <shadow run folder>\n"
            "       uv run python -m prism.retrain train\n"
            '       uv run python -m prism.retrain promote <version> "<what it is>" "<why>"\n'
            '       uv run python -m prism.retrain rollback "<why>"\n'
            "       uv run python -m prism.retrain show"
        )
