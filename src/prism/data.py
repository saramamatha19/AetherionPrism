"""Load the dataset (train.csv / eval.csv). Labels become sets, so order never matters."""
#csv to python objects 
import csv
from dataclasses import dataclass
from pathlib import Path

SOURCES = ("kb", "jira", "confluence", "slack", "notion", "gmail", "web")

HARD_CASE_ORIGINS = ("Hard cases (generated)", "Hard cases (hand-written)")
COMPANY_ORIGIN = "EnterpriseRAG company topics"


@dataclass(frozen=True)
class Question:
    """One row of the dataset."""

    id: str
    text: str
    labels: frozenset[str]  # e.g. {"jira", "confluence"}; empty = no search needed
    question_type: str  # e.g. "Multiple sources"
    part: str  # "real-world", "hard cases" or "company topics"


def parse_labels(raw: str) -> frozenset[str]:
    """Turn "jira|confluence" into {"jira", "confluence"}. Unknown label → error."""
    labels = frozenset(raw.split("|")) if raw else frozenset()
    unknown = labels - set(SOURCES)
    if unknown:
        raise ValueError(f"Unknown label(s): {sorted(unknown)}")
    return labels


def part_of(origin: str) -> str:
    """Which of the 3 parts a row belongs to, based on where it came from."""
    if origin in HARD_CASE_ORIGINS:
        return "hard cases"
    if origin == COMPANY_ORIGIN:
        return "company topics"
    return "real-world"


def load_questions(path: str | Path) -> list[Question]:
    """Read a CSV file into a list of Questions."""
    with open(path, newline="", encoding="utf-8") as f:
        return [
            Question(
                id=row["id"],
                text=row["question"],
                labels=parse_labels(row["labels"]),
                question_type=row["question_type"],
                part=part_of(row["origin"]),
            )
            for row in csv.DictReader(f)
        ]
