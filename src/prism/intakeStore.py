"""The intake store: real questions from the Aetherion platform, kept in ONE SQLite file.

intake/prism.db   one row per question: the question, the platform's intent answer, and (once the
                  model has run) Prism's answer. Never in git: these are real client questions.

This is the ONLY file that knows WHERE questions are kept. The routers and the report just call
save_question, get_unpredicted, save_prediction and get_prediction. Moving to a real database later
means changing only this file.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

STORE_PATH = Path("intake/prism.db")  # in Docker, a volume is linked to the intake/ folder

# One table, made once. PRIMARY KEY = each question_id only once (the duplicate check).
# NOT NULL = must be filled at intake; the last 5 columns stay empty (NULL) until the model runs.
# intent_answer = the platform's source names EXACTLY as sent (translated only when comparing).
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS questions (
    question_id   TEXT PRIMARY KEY,
    question      TEXT NOT NULL,
    intent_answer TEXT NOT NULL,
    received_at   TEXT NOT NULL,
    prism_answer  TEXT,
    model_sure    INTEGER,
    confidence    REAL,
    model_version TEXT,
    predicted_at  TEXT
)
"""


def now() -> str:
    """The time right now, e.g. 2026-10-24T10:15:03"""
    return datetime.now().isoformat(timespec="seconds")


class IntakeStore:
    """ONE store per running program; the lock makes two requests take turns (like DecisionLog)."""

    def __init__(self, path: Path = STORE_PATH):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run(CREATE_TABLE)

    def run(self, sql: str, values: tuple = ()):
        """Run one SQL command, save, close the file. Returns (rows read, how many rows changed)."""
        with self.lock:
            connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row  # read columns by name: row["question"]
            try:
                cursor = connection.execute(sql, values)
                rows = cursor.fetchall()
                changed = cursor.rowcount
                connection.commit()
            finally:
                connection.close()  # always close, even after an error
        return rows, changed

    def save_question(self, question_id: str, question: str, intent_answer: list[str]) -> bool:
        """Save one question from the platform. True = saved, False = already there (skipped)."""
        _, changed = self.run(
            "INSERT OR IGNORE INTO questions (question_id, question, intent_answer, received_at) "
            "VALUES (?, ?, ?, ?)",
            (question_id, question, json.dumps(sorted(intent_answer)), now()),
        )
        return changed == 1

    def get_unpredicted(self) -> list[dict]:
        """The questions the model hasn't answered yet, oldest first."""
        rows, _ = self.run(
            "SELECT question_id, question FROM questions "
            "WHERE prism_answer IS NULL ORDER BY received_at"
        )
        questions = []
        for row in rows:
            questions.append({"question_id": row["question_id"], "question": row["question"]})
        return questions

    def save_prediction(
        self,
        question_id: str,
        prism_answer: list[str],
        model_sure: bool,
        confidence: float,
        model_version: str,
    ) -> bool:
        """Fill in Prism's answer for a saved question. False = no question with that id.
        model_sure = confident enough to answer alone (the go-live check counts only these)."""
        _, changed = self.run(
            "UPDATE questions SET prism_answer = ?, model_sure = ?, confidence = ?, "
            "model_version = ?, predicted_at = ? WHERE question_id = ?",
            (
                json.dumps(sorted(prism_answer)),
                int(model_sure),  # SQLite has no True/False: 1 or 0
                confidence,
                model_version,
                now(),
                question_id,
            ),
        )
        return changed == 1

    def get_prediction(self, question_id: str) -> dict | None:
        """Everything about one question, with lists turned back into lists. None = unknown id.
        prism_answer is None while the model hasn't run on it yet."""
        rows, _ = self.run("SELECT * FROM questions WHERE question_id = ?", (question_id,))
        if not rows:
            return None
        found = dict(rows[0])
        found["intent_answer"] = json.loads(found["intent_answer"])
        if found["prism_answer"] is not None:
            found["prism_answer"] = json.loads(found["prism_answer"])
            found["model_sure"] = found["model_sure"] == 1
        return found
