"""The shadow report (Phase 7): where do the model and the LLM agree or disagree, and what
needs a human look? Reads a shadow-mode decision log; writes a review file for people.

Run:  uv run python -m prism.shadowReport <log folder>              (real questions: you fill in review.csv)
      uv run python -m prism.shadowReport <log folder> rehearsal    (eval: the labels play the human)
      uv run python -m prism.shadowReport <log folder> golive       (reads the filled-in review.csv)
"""

import csv
import random
from pathlib import Path

from prism.data import SOURCES
from prism.decisionLog import read_decisions

REVIEW_COLUMNS = ["kind", "question", "model_answer", "llm_answer", "correct_answer"]


def load_run(log_dir: Path) -> list[dict]:
    """All decisions of one shadow run (every day's JSON file in the folder)."""
    decisions = []
    for path in sorted(Path(log_dir).glob("decisions_*.jsonl")):
        decisions += read_decisions(path)
    return decisions


def shadow_report(decisions, spot_check_rate: float, seed: int) -> dict:
    """Agreement, per-source differences, repeats, and the list of questions a human should check."""
    agree_all = 0
    compared_all = 0
    confident = []
    model_adds = {}
    model_misses = {}
    for source in SOURCES:
        model_adds[source] = 0
        model_misses[source] = 0
    to_review = []
    confident_agreements = []
    seen_texts = {}

    for d in decisions:
        seen_texts[d["question"]] = seen_texts.get(d["question"], 0) + 1
        if d["llm_answer"] is None:  # the LLM failed for this one: nothing to compare
            continue
        model = set(d["model_answer"])
        llm = set(d["llm_answer"])
        compared_all += 1
        agree_all += model == llm

        if d["would_escalate"]:  # not confident: the LLM would still answer after go-live
            continue
        confident.append(d)
        for source in model - llm:
            model_adds[source] += 1
        for source in llm - model:
            model_misses[source] += 1
        if model == llm:
            confident_agreements.append(d)
        else:
            to_review.append({"kind": "disagreement", "decision": d})

    # The case-B spot-check: a random sample of confident AGREEMENTS (repeatable with the seed)
    rng = random.Random(seed)
    sample_size = round(len(confident_agreements) * spot_check_rate)
    for d in rng.sample(confident_agreements, sample_size):
        to_review.append({"kind": "spot-check", "decision": d})

    repeated = 0
    for count in seen_texts.values():
        if count > 1:
            repeated += count - 1

    return {
        "questions": len(decisions),
        "agreement_all": agree_all / compared_all,
        "confident": len(confident),
        "agreement_confident": len(confident_agreements) / len(confident),
        "model_adds": model_adds,
        "model_misses": model_misses,
        "repeat_share": repeated / len(decisions),
        "to_review": to_review,
        "confident_agreements": len(confident_agreements),
    }


def write_review_file(to_review, path: Path, correct_answers=None) -> None:
    """The review file for a HUMAN (open in Excel): fill in correct_answer like "jira|slack"
    (leave it empty only if no search is needed: write "none").
    correct_answers (question → set) is ONLY for the rehearsal, where eval labels play the human."""
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(REVIEW_COLUMNS)
        for item in to_review:
            d = item["decision"]
            correct = ""
            if correct_answers is not None:
                labels = correct_answers[d["question"]]
                if labels:
                    correct = "|".join(sorted(labels))
                else:
                    correct = "none"
            writer.writerow(
                [
                    item["kind"],
                    d["question"],
                    "|".join(d["model_answer"]),
                    "|".join(d["llm_answer"]),
                    correct,
                ]
            )


def read_review_file(path: Path) -> list[dict]:
    """The filled-in review file. An empty correct_answer → ERROR (every row must be checked)."""
    rows = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            if row["correct_answer"] == "":
                raise ValueError(f"Not reviewed yet: {row['question']!r} (fill in correct_answer)")
            rows.append(row)
    return rows


def as_set(text: str) -> frozenset:
    """ "jira|slack" → {"jira", "slack"}; "none" or "" → {} (no search)."""
    if text in ("", "none"):
        return frozenset()
    return frozenset(text.split("|"))


def go_live_check(report: dict, reviewed: list[dict], settings) -> dict:
    """Estimate the model's and the LLM's accuracy on CONFIDENT questions, then GO / NO-GO.

    Agreements: counted as right as often as the spot-check says agreements really are (catches case B).
    Disagreements: each one was checked, so we count exactly who was right.
    """
    spot_checked = 0
    spot_right = 0
    model_right_on_disagreements = 0
    llm_right_on_disagreements = 0
    for row in reviewed:
        correct = as_set(row["correct_answer"])
        if row["kind"] == "spot-check":
            spot_checked += 1
            spot_right += as_set(row["model_answer"]) == correct  # model = LLM here
        else:
            model_right_on_disagreements += as_set(row["model_answer"]) == correct
            llm_right_on_disagreements += as_set(row["llm_answer"]) == correct

    agreement_right_rate = spot_right / spot_checked  # how often agreements are REALLY right
    agreements_right = report["confident_agreements"] * agreement_right_rate

    model_accuracy = (agreements_right + model_right_on_disagreements) / report["confident"]
    llm_accuracy = (agreements_right + llm_right_on_disagreements) / report["confident"]

    checks = {
        f"model accuracy ≥ {settings.go_live_accuracy:.0%}": model_accuracy
        >= settings.go_live_accuracy,
        "model at least as good as the LLM": model_accuracy >= llm_accuracy,
        f"at least {settings.go_live_min_questions} questions": report["questions"]
        >= settings.go_live_min_questions,
    }
    go = True
    for passed in checks.values():
        if not passed:
            go = False
    verdict = "NO-GO"
    if go:
        verdict = "GO"

    return {
        "spot_checked": spot_checked,
        "agreement_right_rate": agreement_right_rate,
        "model_accuracy": model_accuracy,
        "llm_accuracy": llm_accuracy,
        "checks": checks,
        "verdict": verdict,
    }


if __name__ == "__main__":
    import sys

    from prism.config import load_settings
    from prism.data import load_questions

    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "Usage: uv run python -m prism.shadowReport <log folder> [rehearsal | golive]"
        )
    log_dir = Path(sys.argv[1])
    rehearsal = len(sys.argv) == 3 and sys.argv[2] == "rehearsal"
    golive = len(sys.argv) == 3 and sys.argv[2] == "golive"

    settings = load_settings()
    decisions = load_run(log_dir)
    report = shadow_report(decisions, settings.spot_check_rate, settings.seed)

    if golive:
        result = go_live_check(report, read_review_file(log_dir / "review.csv"), settings)
        print(f"Go-live check ({report['questions']} questions, {report['confident']} confident)")
        print(
            f"  spot-check: agreements are really right {result['agreement_right_rate']:.0%} "
            f"of the time ({result['spot_checked']} checked)"
        )
        print(
            f"  model accuracy on confident questions (estimated): {result['model_accuracy']:.1%}"
        )
        print(f"  LLM accuracy on the same questions (estimated):     {result['llm_accuracy']:.1%}")
        for check, passed in result["checks"].items():
            mark = "✅" if passed else "❌"
            print(f"  {mark} {check}")
        print(f"  VERDICT: {result['verdict']}")

        # Rehearsal only: the thermometer test (every eval question has a label → the TRUE accuracy)
        if (log_dir.name).startswith("shadow_eval_replay"):
            labels = {}
            for q in load_questions("data/eval.csv"):
                labels[q.text] = q.labels
            model_true = 0
            llm_true = 0
            confident = 0
            for d in decisions:
                if d["would_escalate"] or d["llm_answer"] is None:
                    continue
                confident += 1
                model_true += frozenset(d["model_answer"]) == labels[d["question"]]
                llm_true += frozenset(d["llm_answer"]) == labels[d["question"]]
            print("\n  Thermometer test (rehearsal: every question labelled → the TRUE accuracy)")
            print(
                f"  model: estimated {result['model_accuracy']:.1%} │ true {model_true / confident:.1%}"
            )
            print(
                f"  LLM:   estimated {result['llm_accuracy']:.1%} │ true {llm_true / confident:.1%}"
            )
        raise SystemExit(0)

    print(
        f"Shadow report: {report['questions']} questions │ confident: {report['confident']} "
        f"({report['confident'] / report['questions']:.0%})"
    )
    print(f"  agreement, all questions:        {report['agreement_all']:.1%}")
    print(f"  agreement, CONFIDENT questions:  {report['agreement_confident']:.1%}")
    print(f"\n  {'per source (confident)':<24}{'model adds':>11}{'model misses':>14}")
    for source in SOURCES:
        print(
            f"  {source:<24}{report['model_adds'][source]:>11}{report['model_misses'][source]:>14}"
        )
    print(f"\n  repeats (same question asked again): {report['repeat_share']:.1%}")

    correct_answers = None
    if rehearsal:
        correct_answers = {}
        for q in load_questions("data/eval.csv"):
            correct_answers[q.text] = q.labels
    review_path = log_dir / "review.csv"
    write_review_file(report["to_review"], review_path, correct_answers)
    disagreements = 0
    for item in report["to_review"]:
        disagreements += item["kind"] == "disagreement"
    spot_checks = len(report["to_review"]) - disagreements
    print(
        f"\nReview file: {disagreements} disagreements + {spot_checks} spot-checks → {review_path}"
    )
    if rehearsal:
        print("  (rehearsal: correct_answer filled from the eval labels)")
    else:
        print(
            "  → fill in the correct_answer column (e.g. jira|slack, or none), then run the go-live check"
        )
