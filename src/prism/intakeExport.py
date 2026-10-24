"""The intake export: prism.db → a decision log, so the EXISTING shadow tools can evaluate it.

The platform's intent prompt plays the "LLM" in those tools:
    prism_answer → model_answer · intent_answer (translated to our names) → llm_answer
    not model_sure → would_escalate · mode = "shadow"

Run:  uv run python -m prism.intakeExport                         (all predicted questions so far)
      uv run python -m prism.intakeExport --since 2026-11-10      (only questions from that day on)
      uv run python -m prism.intakeExport --db <copy of prism.db>  (e.g. a copy from the server)
Then: uv run python -m prism.shadowReport logs/platform_<date>    (and golive, retrain collect: as before)

Not exported, but counted: questions the model hasn't answered yet (waiting), and questions whose
platform source names aren't in platform_source_names (sources Prism doesn't know, e.g. sharepoint).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from prism.config import load_settings
from prism.data import SOURCES
from prism.intakeStore import STORE_PATH, IntakeStore

LOGS_DIR = Path("logs")


def translate(platform_names: list[str], table: dict) -> list[str] | None:
    """His source names → ours. None = at least one name isn't in the table."""
    ours = []
    for name in platform_names:
        if name not in table:
            return None
        ours.append(table[name])
    return sorted(set(ours))


def export(store, table: dict, out_dir: Path, since: str | None = None) -> dict:
    """Write the decision log into out_dir (a new folder) and return what happened."""
    for name in table.values():
        if name not in SOURCES:
            raise ValueError(
                f"platform_source_names maps to {name!r}, which isn't one of {SOURCES}"
            )
    if out_dir.exists():
        raise ValueError(f"{out_dir} already exists: an export is never overwritten")

    decisions = []
    skipped_unknown = 0
    unknown_names = {}  # name → in how many questions
    for row in store.get_predicted(since):
        llm_answer = translate(row["intent_answer"], table)
        if llm_answer is None:
            skipped_unknown += 1
            for name in row["intent_answer"]:
                if name not in table:
                    unknown_names[name] = unknown_names.get(name, 0) + 1
            continue
        decisions.append(
            {
                "question_id": row["question_id"],
                "time": row["received_at"],
                "question": row["question"],
                "model_answer": row["prism_answer"],
                "llm_answer": llm_answer,  # the platform's intent prompt, in our names
                "would_escalate": not row["model_sure"],
                "model_confidence": row["confidence"],
                "mode": "shadow",
                "model_version": row["model_version"],
                "llm_model": "platform intent prompt",
            }
        )

    if decisions:  # nothing to compare → no folder (the shadow report needs at least 1 question)
        out_dir.mkdir(parents=True)
        with open(out_dir / "decisions_platform.jsonl", "w") as file:
            for decision in decisions:
                file.write(json.dumps(decision) + "\n")

    return {
        "exported": len(decisions),
        "waiting": len(store.get_unpredicted()),
        "skipped_unknown": skipped_unknown,
        "unknown_names": unknown_names,
        "folder": str(out_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prism.db → a decision log for the shadow tools")
    parser.add_argument(
        "--since", help="only questions received on or after this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--db", default=str(STORE_PATH), help="which prism.db (default: intake/prism.db)"
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"No database at {args.db}")
    settings = load_settings()
    out_dir = LOGS_DIR / datetime.now().strftime("platform_%Y-%m-%d_%H%M")
    summary = export(IntakeStore(args.db), settings.platform_source_names, out_dir, args.since)

    if summary["exported"] == 0:
        print("Nothing to export: no predicted questions with known source names.")
    else:
        print(f"Exported {summary['exported']} questions → {summary['folder']}/")
    print(f"  waiting (not predicted yet):       {summary['waiting']}")
    print(f"  skipped, unknown platform names:   {summary['skipped_unknown']}")
    for name, count in summary["unknown_names"].items():
        print(f"    {name}: {count} questions")
    if summary["exported"]:
        print(f"Next: uv run python -m prism.shadowReport {summary['folder']}")
