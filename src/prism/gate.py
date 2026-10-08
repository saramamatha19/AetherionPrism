"""The gate: the confidence line above which the model answers by itself."""

# The gate values we try: 0.00, 0.01, 0.02, ... 0.99
GATE_CANDIDATES = []
for i in range(0, 100):
    GATE_CANDIDATES.append(round(0.01 * i, 2))


def coverage_at(confidences: list[float], correct: list[bool], gate: float) -> dict:
    """At one gate: how many questions the model keeps, and how often it's right on those."""
    kept = 0
    kept_right = 0
    for i in range(len(confidences)):
        if confidences[i] >= gate:
            kept += 1
            if correct[i]:
                kept_right += 1

    if kept == 0:
        accuracy = 0.0  # nothing kept → nothing to be right about
    else:
        accuracy = kept_right / kept

    return {
        "gate": gate,
        "kept": kept,
        "coverage": kept / len(confidences),
        "accuracy": accuracy,
    }


def risk_coverage(confidences: list[float], correct: list[bool]) -> list[dict]:
    """For every candidate gate: how many questions the model keeps, and how often it's right on those."""
    table = []
    for gate in GATE_CANDIDATES:
        row = coverage_at(confidences, correct, gate)
        if row["kept"] == 0:  # higher gates would keep nothing either → stop
            break
        table.append(row)
    return table


def choose_gate(table: list[dict], accuracy_floor: float) -> float:
    """The lowest gate where the kept questions are right at least `accuracy_floor` of the time."""
    for row in table:  # the table goes from the lowest gate to the highest
        if row["accuracy"] >= accuracy_floor:
            gate = row["gate"]
            if gate == 0:
                raise ValueError(
                    "Gate is 0: the model would answer everything and never ask the LLM. Refusing."
                )
            return gate

    raise ValueError(
        f"No gate reaches {accuracy_floor:.0%} accuracy: the model is never good enough on its own."
    )
