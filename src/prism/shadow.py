"""Shadow mode on a set of questions (Phase 7).

Rehearsal: the eval questions go through the REAL router in shadow mode. The LLM answers come
from the recording (free, and identical to what gpt-4o answered live), so the whole shadow
process can be tested on questions whose right answers we already know.

Run:  uv run python -m prism.shadow eval
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from prism.decisionLog import DecisionLog
from prism.llm_answer_eval import answers_file, read_answers
from prism.router import Router


class ReplayLLMClient:
    """Pretends to be the OpenAI client, but answers from a recording instead of calling the API.

    The router can't tell the difference: it gets a normal-looking reply, with the REAL
    recorded token counts, so costs in the log are real too.
    """

    def __init__(self, questions, settings):
        recorded_by_id = read_answers(answers_file(settings, "eval"))
        self.recorded_by_text = {}
        for q in questions:
            self.recorded_by_text[q.text] = recorded_by_id[q.id]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        question_text = kwargs["messages"][-1]["content"]  # the user message = the question

        if question_text == "warm-up":  # the router's start-up call (never logged): no recording
            content = json.dumps({"sources": []})
            tokens_in = 0
            cached = 0
            tokens_out = 0
        else:
            record = self.recorded_by_text[question_text]  # not recorded → error (never a guess)
            content = json.dumps({"sources": record["llm_sources"]})
            tokens_in = record["prompt_tokens"]
            cached = record["cached_tokens"]
            tokens_out = record["completion_tokens"]

        message = SimpleNamespace(content=content, refusal=None)
        usage = SimpleNamespace(
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def run_shadow(questions, settings, log_dir: Path) -> Path:
    """Send every question through the real router in SHADOW mode; return the log folder."""
    shadow_settings = settings.model_copy(update={"mode": "shadow"})
    router = Router(
        settings=shadow_settings,
        llm_client=ReplayLLMClient(questions, settings),
        log=DecisionLog(logs_dir=log_dir),
    )
    for i, q in enumerate(questions):
        router.route(q.text)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1} questions …")
    return log_dir


if __name__ == "__main__":
    import sys

    from prism.config import load_settings
    from prism.decisionLog import read_decisions
    from prism.data import load_questions

    if len(sys.argv) != 2 or sys.argv[1] != "eval":
        raise SystemExit("Usage: uv run python -m prism.shadow eval")

    settings = load_settings()
    questions = load_questions("data/eval.csv")
    run_name = datetime.now().strftime("shadow_eval_replay_%Y-%m-%d_%H%M")
    log_dir = Path("logs") / run_name  # a NEW folder per run: runs never mix
    print(f"Shadow rehearsal: {len(questions)} eval questions → {log_dir}/")
    run_shadow(questions, settings, log_dir)

    decisions = []
    for path in sorted(log_dir.glob("decisions_*.jsonl")):
        decisions += read_decisions(path)
    total_cost = 0.0
    for d in decisions:
        total_cost += d["cost_spent"]
    print(f"Done: {len(decisions)} decisions logged │ LLM cost (recorded, real): ${total_cost:.2f}")
