"""Ask the LLM a set of questions ONCE, save the answers, and replay them for free (Phase 5).

Run:  uv run python -m prism.llm_answer_eval        (records the eval set)
Saved: recordings/llm_answer_<set>_<model>_<prompt fingerprint>.jsonl  (one line per question, in git)
       <set> = "eval" (the final exam) or "practice" (train questions for improving the prompt)
"""

import hashlib
import json
from pathlib import Path

from prism.llm import ask_llm_safely, make_client, system_prompt

RECORDINGS_DIR = Path("recordings")


def prompt_fingerprint(prompt: str) -> str:
    """A short code computed from the prompt. Change one word of the prompt → a different code."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def answers_file(
    settings, question_set: str = "eval", recordings_dir: Path = RECORDINGS_DIR
) -> Path:
    """Where the LLM's answers are saved: one file per question set + model + prompt combination."""
    fingerprint = prompt_fingerprint(system_prompt(settings))
    name = f"llm_answer_{question_set}_{settings.llm_model}_{fingerprint}.jsonl"
    return recordings_dir / name


def read_answers(path: Path) -> dict:
    """Read the file → {question id: saved answer}. Later lines win (a retry replaces a failure)."""
    answers = {}
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            answers[record["id"]] = record
    return answers


def record_llm_answers(
    questions, settings, question_set="eval", client=None, recordings_dir: Path = RECORDINGS_DIR
) -> dict:
    """Ask the LLM every question that isn't saved yet (or failed last time). Append each answer.

    `client` is only passed in by tests (a fake LLM); normally the real OpenAI client is made here.
    """
    path = answers_file(settings, question_set, recordings_dir)
    path.parent.mkdir(exist_ok=True)
    saved = read_answers(path)
    if client is None:
        client = make_client(settings)

    asked = 0
    failed = 0
    with open(path, "a") as file:  # "a" = append: never wipe what's already saved
        for q in questions:
            already = saved.get(q.id)
            if already is not None and already["llm_error"] is None:
                continue  # answered successfully before → don't pay again

            result = ask_llm_safely(client, settings, q.text)  # no model answer during recording

            llm_sources = None
            if result["llm_sources"] is not None:
                llm_sources = sorted(result["llm_sources"])  # tidy, same order every run

            record = {
                "id": q.id,
                "llm_sources": llm_sources,
                "llm_error": result["llm_error"],
                "prompt_tokens": result["prompt_tokens"],
                "cached_tokens": result["cached_tokens"],
                "completion_tokens": result["completion_tokens"],
                "latency_ms": result["latency_ms"],
            }
            file.write(json.dumps(record) + "\n")
            file.flush()  # save right away, so a crash loses at most one question

            asked += 1
            if result["llm_error"] is not None:
                failed += 1
            if asked % 50 == 0:
                print(f"  asked {asked} …")

    return {"file": str(path), "asked_now": asked, "failed_now": failed}


def load_llm_answers(
    questions, settings, question_set="eval", recordings_dir: Path = RECORDINGS_DIR
) -> dict:
    """Replay: {question id: LLM sources}. Any question missing or failed → ERROR, never "search nothing"."""
    path = answers_file(settings, question_set, recordings_dir)
    saved = read_answers(path)
    answers = {}
    missing = []
    for q in questions:
        record = saved.get(q.id)
        if record is None or record["llm_error"] is not None:
            missing.append(q.id)
        else:
            answers[q.id] = frozenset(record["llm_sources"])
    if missing:
        raise ValueError(
            f"{len(missing)} questions have no LLM answer in {path} (e.g. {missing[:3]}). "
            "Run: uv run python -m prism.llm_answer_eval"
        )
    return answers


if __name__ == "__main__":
    from prism.config import load_settings
    from prism.data import load_questions

    settings = load_settings()
    questions = load_questions("data/eval.csv")
    print(f"Recording {settings.llm_model} answers for {len(questions)} eval questions")
    print(f"File: {answers_file(settings)}")
    summary = record_llm_answers(questions, settings)
    print(f"\nAsked now: {summary['asked_now']}, failed now: {summary['failed_now']}")
    if summary["failed_now"]:
        print("Some failed → run the same command again: it retries only those.")
