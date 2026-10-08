"""How sure the model is about its whole answer: its least sure judge (0 = coin flip, 1 = certain)."""


def how_sure(score: float, threshold: float) -> float:
    """One judge: how far its score is from its threshold, as a share of the room on that side."""
    if score >= threshold:
        room = 1 - threshold  # room on the "yes" side
        distance = score - threshold
    else:
        room = threshold  # room on the "no" side
        distance = threshold - score
    return distance / room


def confidence(scores: dict[str, float], thresholds: dict[str, float]) -> tuple[float, str]:
    """The whole answer is only as sure as its least sure judge. Returns (confidence, that source)."""
    lowest = None
    weakest_source = None

    for source, score in scores.items():
        sure = how_sure(score, thresholds[source])
        if lowest is None or sure < lowest:
            lowest = sure
            weakest_source = source

    return lowest, weakest_source
