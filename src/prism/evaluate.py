"""Grade any "student" on the eval set: marks by part, by question type and by source."""

from collections.abc import Callable

from prism.data import Question
from prism.metrics import exact_match, found_all_needed, per_source_scores

# A "student" = any function: question text → set of sources
Student = Callable[[str], frozenset[str]]


def score_group(truths: list[frozenset[str]], predictions: list[frozenset[str]]) -> dict:
    """Turn True/False marks into percentages for one group of questions."""
    pairs = list(zip(truths, predictions))
    return {
        "questions": len(pairs),
        "exact_match": sum(exact_match(t, p) for t, p in pairs) / len(pairs),
        "found_all_needed": sum(found_all_needed(t, p) for t, p in pairs) / len(pairs),
    }


def score_by(field: str, questions: list[Question], predictions: list[frozenset[str]]) -> dict:
    """Score each group separately, e.g. field="part" → real-world, hard cases, company topics."""
    groups = sorted({getattr(q, field) for q in questions})
    result = {}
    for name in groups:
        in_group = [i for i, q in enumerate(questions) if getattr(q, field) == name]
        result[name] = score_group(
            [questions[i].labels for i in in_group], [predictions[i] for i in in_group]
        )
    return result


def evaluate(student: Student, questions: list[Question]) -> dict:
    """Ask the student every question, then mark the answers."""
    predictions = [student(q.text) for q in questions]
    truths = [q.labels for q in questions]
    return {
        "all": score_group(truths, predictions),
        "by_part": score_by("part", questions, predictions),
        "by_type": score_by("question_type", questions, predictions),
        "by_source": per_source_scores(truths, predictions),
    }


def print_table(title: str, groups: dict) -> None:
    """Print one table: a row per group with questions, exact match and all-needed-found."""
    print(f"\n{title:<22}{'Questions':>10}{'Exact':>9}{'All needed':>12}")
    for name, s in groups.items():
        print(f"{name:<22}{s['questions']:>10}{s['exact_match']:>9.1%}{s['found_all_needed']:>12.1%}")


def print_report(student_name: str, result: dict) -> None:
    """Print the full report: by part, by question type, and by source."""
    print(f"\n===== Student: {student_name} =====")
    print_table("By part", {**result["by_part"], "ALL": result["all"]})
    print_table("By question type", result["by_type"])

    print(f"\n{'By source':<22}{'Precision':>10}{'Recall':>9}{'F1':>12}")
    for source, s in result["by_source"].items():
        print(f"{source:<22}{s['precision']:>10.1%}{s['recall']:>9.1%}{s['f1']:>12.1%}")
