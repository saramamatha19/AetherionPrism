"""Marking rules: compare predicted sources with the answer key (both are sets)."""

from prism.data import SOURCES


def exact_match(truth: frozenset[str], predicted: frozenset[str]) -> bool:
    """Everything right, nothing extra."""
    return truth == predicted


def found_all_needed(truth: frozenset[str], predicted: frozenset[str]) -> bool:
    """Every needed source was picked (extra ones are allowed)."""
    return truth <= predicted


def per_source_scores(
    truths: list[frozenset[str]], predictions: list[frozenset[str]]
) -> dict[str, dict[str, float]]:
    """Precision, recall and F1 for each of the 7 sources."""
    scores = {}
    for source in SOURCES:
        pairs = list(zip(truths, predictions))
        hit = sum(source in t and source in p for t, p in pairs)  # picked and needed
        extra = sum(source not in t and source in p for t, p in pairs)  # picked, not needed
        missed = sum(source in t and source not in p for t, p in pairs)  # needed, not picked

        precision = hit / (hit + extra) if hit + extra else 0.0
        recall = hit / (hit + missed) if hit + missed else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[source] = {"precision": precision, "recall": recall, "f1": f1}
    return scores
