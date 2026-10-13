"""The live router (Phase 6): router.route(question) → a DecisionRecord, in 3 modes.

model_only: the model always answers; the LLM is never called (for testing, no cost, no API key).
shadow:     the LLM always answers the user; the model's answer is only RECORDED (the trial period).
gated:      the model answers when sure; the LLM otherwise (+ negation rule) (live, after shadow).

Try it:  uv run python -m prism.router "is the login bug fixed?"   (uses the mode in the YAML)
"""

import os
import random
import time
from datetime import datetime

from prism.confidence import confidence
from prism.config import load_settings
from prism.cost import llm_call_cost
from prism.decisionRecord import DecisionRecord
from prism.llm import ask_llm_safely, make_client
from prism.llm_answer_eval import answers_file, read_answers
from prism.logreg import scores
from prism.registry import data_fingerprint, load_card, load_model, read_champion
from prism.ruleNegation import has_negation
from prism.system_eval import route_one
from prism.thresholds import apply_thresholds


def average_recorded_llm_cost(settings) -> float:
    """The average REAL cost of one LLM call, from the eval recording of this model + prompt.
    Used as "what this question would have cost" when the LLM wasn't asked (the savings counter)."""
    path = answers_file(settings, "eval")
    records = read_answers(path)
    if not records:
        raise ValueError(f"No LLM recording at {path}: run uv run python -m prism.llm_answer_eval")
    total = 0.0
    for record in records.values():
        total += llm_call_cost(
            record["prompt_tokens"], record["cached_tokens"], record["completion_tokens"], settings
        )
    return total / len(records)


class Router:
    """Loads everything ONCE (a few seconds), then route(question) is fast."""

    def __init__(self, settings=None, llm_client=None, log=None):
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self.mode = settings.mode
        self.model_version = read_champion()["version"]  # the live model: models/champion.json
        self.model, self.thresholds, self.gate = load_model(self.model_version)
        card_settings = load_card(self.model_version)["settings"]
        self.negation_rule_on = card_settings.get("negation_to_llm", False)
        self.config_hash = data_fingerprint("configs/default.yaml")  # exactly which settings
        self.audit_draw = random.Random(settings.seed)  # repeatable audit draws
        self.log = log  # the decision log (Step 6.3); None = don't write

        # The LLM client: needed unless EVERY question is model_only. Each question can choose its
        # own mode (the UI), so it's made whenever an API key exists; without a key, model_only works.
        self.llm_client = llm_client
        key_available = os.environ.get("OPENAI_API_KEY") is not None
        if self.llm_client is None and (self.mode != "model_only" or key_available):
            self.llm_client = make_client(settings)

        # "What the LLM would have cost": the average REAL cost per question from the recording
        self.average_llm_cost = average_recorded_llm_cost(settings)

        # The first call builds BGE (~1 s); do it now so the first real user doesn't wait.
        # It is NEVER logged: a fake question must not look like real traffic (prototype mistake #24).
        self.route("warm-up", write_log=False)

    def route(
        self, question_text: str, write_log: bool = True, mode: str | None = None
    ) -> DecisionRecord:
        """Decide where to search for ONE question → a DecisionRecord (and write it to the log).

        mode: this question's own mode (e.g. chosen in the UI). None → the router's mode (YAML).
        It never changes the router's mode, so one user's choice can't affect anyone else.
        """
        if mode is None:
            mode = self.mode
        start = time.perf_counter()

        # 1. The model ALWAYS runs first (fast, free), so its answer is always recorded
        model_scores = scores(self.model, question_text)
        model_answer = apply_thresholds(model_scores, self.thresholds)
        model_confidence, _ = confidence(model_scores, self.thresholds)

        # 2. What would GATED mode do? (the same rule as Phase 5.6)
        gated_choice = route_one(
            model_confidence, self.gate, has_negation(question_text), self.negation_rule_on
        )
        would_escalate = gated_choice != "model"

        # 3. Decide, depending on the mode
        ask_llm = False
        if mode == "model_only":
            reason = "model only"
        elif mode == "shadow":
            ask_llm = True
            reason = "shadow mode"
        elif would_escalate:  # gated: the negation rule, or the model is unsure
            ask_llm = True
            if gated_choice == "llm (negation rule)":
                reason = "negation rule"
            else:
                reason = "model unsure"
        elif self.audit_draw.random() < self.settings.audit_rate:  # gated: a random spot-check
            ask_llm = True
            reason = "audit"
        else:
            reason = "model sure"

        # 4. Ask the LLM if needed (errors never crash: the backup plan is the model's answer)
        answer = model_answer
        decided_by = "model"
        llm_answer = None
        llm_error = None
        cost_spent = 0.0
        if ask_llm:
            if self.llm_client is None:
                raise ValueError(
                    f"Mode {mode!r} needs the LLM, but there's no OPENAI_API_KEY in .env"
                )
            result = ask_llm_safely(self.llm_client, self.settings, question_text, model_answer)
            llm_answer = result["llm_sources"]
            llm_error = result["llm_error"]
            cost_spent = llm_call_cost(
                result["prompt_tokens"],
                result["cached_tokens"],
                result["completion_tokens"],
                self.settings,
            )
            if llm_error is not None:
                reason = "llm failed → backup"
                answer = result["sources"]  # the backup plan (model answer, or all 7)
            elif reason != "audit":  # audit: the model still answers; the LLM is only recorded
                answer = llm_answer
                decided_by = "llm"

        if ask_llm:
            cost_if_llm = cost_spent
        else:
            cost_if_llm = self.average_llm_cost

        # 5. The receipt
        record = DecisionRecord(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question=question_text,
            answer=frozenset(answer),
            decided_by=decided_by,
            reason=reason,
            model_answer=model_answer,
            model_confidence=model_confidence,
            gate=self.gate,
            would_escalate=would_escalate,
            llm_answer=llm_answer,
            llm_error=llm_error,
            cost_spent=cost_spent,
            cost_if_llm=cost_if_llm,
            latency_ms=(time.perf_counter() - start) * 1000,
            mode=mode,  # this question's own mode
            model_version=self.model_version,
            llm_model=self.settings.llm_model,
            prompt_version=self.settings.llm_prompt_version,
            config_hash=self.config_hash,
        )

        # 6. Write it to the decision log (Step 6.3), except the warm-up
        if write_log and self.log is not None:
            self.log.write(record)
        return record


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit('Usage: uv run python -m prism.router "your question" ["another" …]')
    from prism.decisionLog import DecisionLog

    router = Router(log=DecisionLog())
    print(f"mode: {router.mode} │ model: {router.model_version} │ gate: {router.gate}")
    for text in sys.argv[1:]:
        r = router.route(text)
        print(
            f"\n{text}\n  → {sorted(r.answer)}  decided by {r.decided_by} ({r.reason})"
            f" │ model {sorted(r.model_answer)} conf {r.model_confidence:.2f}"
            f" │ ${r.cost_spent:.5f} │ {r.latency_ms:.0f} ms"
        )
