"""The decision log (Phase 6): every REAL decision → one JSON line (for code) + one CSV row (for people).

logs/decisions_<date>.jsonl   read by code (Phase 7 shadow report, Phase 8 retraining)
logs/decisions_<date>.csv     for reading and debugging: open it in Excel / Numbers
A new pair of files each day; logs/ is never in git. The router's warm-up is never written.
"""

import csv
import json
import threading
from datetime import date
from pathlib import Path

from prism.decisionRecord import CSV_COLUMNS

LOGS_DIR = Path("logs")


class DecisionLog:
    """ONE writer per running program, so lines never get mixed (prototype mistake #9)."""

    def __init__(self, logs_dir: Path = LOGS_DIR):
        self.logs_dir = logs_dir
        self.lock = threading.Lock()  # two questions at the same moment → they take turns writing

    def files_for(self, day: str):
        """The day's two files, e.g. logs/decisions_2026-10-13.jsonl and .csv"""
        json_path = self.logs_dir / f"decisions_{day}.jsonl"
        csv_path = self.logs_dir / f"decisions_{day}.csv"
        return json_path, csv_path

    def write(self, record) -> None:
        """Add one decision to today's JSON file and today's CSV file."""
        with self.lock:
            self.logs_dir.mkdir(exist_ok=True)
            json_path, csv_path = self.files_for(date.today().isoformat())

            with open(json_path, "a") as file:  # "a" = append: never wipe earlier decisions
                file.write(record.to_json() + "\n")

            is_new_csv = not csv_path.exists()
            with open(csv_path, "a", newline="") as file:
                writer = csv.writer(file)
                if is_new_csv:
                    writer.writerow(CSV_COLUMNS)  # the header row, once per day's file
                writer.writerow(record.to_csv_row())


def read_decisions(path: Path) -> list[dict]:
    """Read a day's JSON lines back, with exact values (Phase 7 uses this, never the CSV)."""
    decisions = []
    for line in Path(path).read_text().splitlines():
        decisions.append(json.loads(line))
    return decisions


def count_shadow_questions(logs_dir: Path = LOGS_DIR) -> int:
    """How many SHADOW-mode questions are in the day logs (logs/decisions_*.jsonl): the progress
    towards the go-live check. Rehearsal runs live in sub-folders, so they're not counted."""
    count = 0
    for path in sorted(Path(logs_dir).glob("decisions_*.jsonl")):
        for decision in read_decisions(path):
            if decision["mode"] == "shadow":
                count += 1
    return count
