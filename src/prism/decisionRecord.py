"""The "receipt" for every routing decision: what was answered, who decided, why, and at what cost.

One shape everywhere: the router fills it in, the decision log writes it (JSON lines for code,
CSV for people), and Phase 7 (shadow mode) reads it back.
"""

import json
from dataclasses import dataclass

# The CSV columns, in this order (the header row is written once, when a day's file is new).
# The JSON uses the same fields, so the two files always match.
CSV_COLUMNS = [
    "time",
    "question",
    "answer",
    "decided_by",
    "reason",
    "model_answer",
    "model_confidence",
    "gate",
    "would_escalate",
    "llm_answer",
    "llm_error",
    "cost_spent",
    "cost_if_llm",
    "latency_ms",
    "mode",
    "model_version",
    "llm_model",
    "prompt_version",
    "config_hash",
]


def sources_text(sources) -> str:
    """{"slack", "jira"} → "jira|slack" (sorted, joined like our dataset); None → ""."""
    if sources is None:
        return ""
    return "|".join(sorted(sources))


@dataclass
class DecisionRecord:
    """Everything about ONE routing decision."""

    time: str  # when, e.g. "2026-10-13 14:02:11"
    question: str  # what the user asked
    answer: frozenset  # the sources the user gets
    decided_by: str  # "model" or "llm"
    reason: str  # "model sure" | "model unsure" | "negation rule" | "shadow mode"
    #              | "model only" | "audit" | "llm failed → backup"
    model_answer: frozenset  # ALWAYS kept, even when the LLM answered (plan B8)
    model_confidence: float
    gate: float  # the bar the model had to pass
    would_escalate: bool  # would GATED mode have asked the LLM? (needed in shadow mode)
    llm_answer: frozenset | None  # what the LLM said (None if not asked, or if it failed)
    llm_error: str | None  # why the LLM failed (None if fine, or not asked)
    cost_spent: float  # what this question really cost, in $
    cost_if_llm: float  # what it would cost if the LLM answered (to show the savings)
    latency_ms: float  # how long the whole decision took
    mode: str  # "model_only" | "shadow" | "gated"
    model_version: str  # e.g. "tfidf_ftv2_bge_logreg_v2"
    llm_model: str  # e.g. "gpt-4o-2024-11-20"
    prompt_version: str  # e.g. "v4"
    config_hash: str  # fingerprint of configs/default.yaml: exactly which settings

    def money_saved(self) -> float:
        """What this question saved compared with asking the LLM every time ($)."""
        return self.cost_if_llm - self.cost_spent

    def model_and_llm_agree(self) -> bool | None:
        """Did the model and the LLM give the same sources? None if the LLM wasn't asked."""
        if self.llm_answer is None:
            return None
        return set(self.model_answer) == set(self.llm_answer)

    def to_dict(self) -> dict:
        """The record as plain values: sets become sorted lists, so JSON can store them."""
        values = {}
        for name in CSV_COLUMNS:
            value = getattr(self, name)
            if isinstance(value, frozenset):
                value = sorted(value)
            values[name] = value
        return values

    def to_json(self) -> str:
        """One JSON line, for CODE: every value comes back exactly (lists, None, numbers, True/False)."""
        return json.dumps(self.to_dict())

    def to_csv_row(self) -> list[str]:
        """One CSV row, for PEOPLE: lists joined with "|", empty for None, numbers rounded."""
        row = []
        for name in CSV_COLUMNS:
            value = getattr(self, name)
            if name in ("answer", "model_answer", "llm_answer"):
                row.append(sources_text(value))
            elif value is None:
                row.append("")
            elif name in ("model_confidence", "gate"):
                row.append(f"{value:.2f}")
            elif name in ("cost_spent", "cost_if_llm"):
                row.append(f"{value:.5f}")
            elif name == "latency_ms":
                row.append(f"{value:.1f}")
            else:
                row.append(str(value))
        return row
