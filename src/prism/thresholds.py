"""Each source's own pass mark, chosen on the threshold slice (never on eval)."""

from prism.metrics import per_source_scores

# The pass marks we try: 0.05, 0.10, 0.15, ... 0.95
CANDIDATES = []
for i in range(1, 20):
    CANDIDATES.append(round(0.05 * i, 2))


def choose_thresholds(
    all_scores: list[dict[str, float]], truths: list[frozenset[str]]
) -> dict[str, float]:
    """For each source, try every candidate pass mark and keep the one with the best F1."""
    thresholds = {}

    for source in all_scores[0]:
        best_f1 = -1.0  # worse than anything, so the first mark tried becomes the first "best"
        best_mark = None  # no hidden default like 0.5

        for mark in CANDIDATES:
            # Pretend this source is the only one: passes the mark → {source}, else → {}
            predictions = []
            for s in all_scores:
                if s[source] >= mark:
                    predictions.append(frozenset({source}))
                else:
                    predictions.append(frozenset())

            f1 = per_source_scores(truths, predictions)[source]["f1"]

            if f1 > best_f1:  # strictly better → on a tie, the lower mark wins
                best_f1 = f1
                best_mark = mark

        thresholds[source] = best_mark

    return thresholds


def apply_thresholds(scores: dict[str, float], thresholds: dict[str, float]) -> frozenset[str]:
    """{"jira": 0.59, "slack": 0.12, ...} → {"jira"}: the sources that pass their own mark."""
    passed = []
    for source, score in scores.items():
        if score >= thresholds[source]:
            passed.append(source)
    return frozenset(passed)
