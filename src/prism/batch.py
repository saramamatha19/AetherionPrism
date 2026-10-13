"""Many questions at once: split pasted text or an uploaded file into questions, then route each.

The page's Batch tab only shows what these functions return. Every question goes through the
normal router.route(), so each one is logged like a single question (and, in shadow mode,
can later become training data through `retrain collect`).
"""

import io
from pathlib import Path

import pandas as pd

from prism.decisionRecord import sources_text

FILE_TYPES = ["txt", "csv", "xlsx"]


def _keep(piece: str, questions: list, seen: set) -> None:
    """Tidy one piece and add it, unless it is empty, only punctuation, or a repeat."""
    clean = " ".join(piece.split())  # remove extra spaces and tabs
    has_letter_or_digit = False
    for char in clean:
        if char.isalnum():
            has_letter_or_digit = True
            break
    if not has_letter_or_digit:
        return
    if clean.lower() in seen:  # the same question twice → keep the first
        return
    seen.add(clean.lower())
    questions.append(clean)


def split_questions(text: str) -> list:
    """Pasted text → questions.

    1. every line is split on its own
    2. inside a line, a question ends after "?" or "!", or after "." followed by a space
       ("Is the bug fixed? What did slack say." → 2 questions; "v1.2" is not split)
    3. empty pieces and repeats are dropped
    A best guess for paragraphs: the page shows the result before anything is routed.
    """
    questions = []
    seen = set()
    for line in text.splitlines():
        piece = ""
        for i in range(len(line)):
            char = line[i]
            piece = piece + char
            ends_here = False
            if char == "?" or char == "!":
                ends_here = True
            if char == ".":
                is_last_char = i + 1 == len(line)
                if is_last_char or line[i + 1] == " ":
                    ends_here = True
            if ends_here:
                _keep(piece, questions, seen)
                piece = ""
        _keep(piece, questions, seen)  # whatever is left at the end of the line
    return questions


def _question_column(table: pd.DataFrame):
    """The column called "question" (any case), or None if there isn't one."""
    for name in table.columns:
        if str(name).strip().lower() == "question":
            return name
    return None


def _cells_to_questions(read_table, data: bytes) -> list:
    """A CSV/Excel file → questions: the "question" column if there is one, else the first column."""
    table = read_table(io.BytesIO(data))
    column = _question_column(table)
    if column is None:  # no header → the first row is a question too, so read again without one
        table = read_table(io.BytesIO(data), header=None)
        column = table.columns[0]
    cells = []
    for value in table[column]:
        if pd.isna(value):  # empty cell
            continue
        cells.append(str(value))
    return split_questions("\n".join(cells))  # a cell may hold a paragraph too


def read_questions_file(file_name: str, data: bytes) -> list:
    """An uploaded file → questions. .txt (lines/paragraphs), .csv or .xlsx (one per row)."""
    file_type = Path(file_name).suffix.lower().lstrip(".")
    if file_type == "txt":
        return split_questions(data.decode("utf-8-sig"))  # utf-8-sig: ignores Excel's BOM mark
    if file_type == "csv":
        return _cells_to_questions(pd.read_csv, data)
    if file_type == "xlsx":
        return _cells_to_questions(pd.read_excel, data)
    raise ValueError(f"Can't read '{file_name}': use one of {', '.join(FILE_TYPES)}")


def route_many(router, questions: list, max_questions: int, mode=None, on_progress=None) -> list:
    """Route each question with the normal router (each one logged) → DecisionRecords, in order.

    max_questions (batch_max_questions in the YAML) stops a huge file from running up a bill.
    on_progress(done, total) is called after each question (the page moves its progress bar).
    """
    if len(questions) > max_questions:
        raise ValueError(
            f"{len(questions)} questions is more than the limit of {max_questions} "
            "(batch_max_questions in the YAML). Split the file."
        )
    records = []
    for i in range(len(questions)):
        records.append(router.route(questions[i], mode=mode))
        if on_progress is not None:
            on_progress(i + 1, len(questions))
    return records


def results_rows(records) -> list:
    """Records → rows for the results table and the downloaded CSV (formatting only)."""
    rows = []
    for record in records:
        search_in = sources_text(record.answer)
        if search_in == "":
            search_in = "no search"
        rows.append(
            {
                "question": record.question,
                "search in": search_in,
                "decided by": record.decided_by,
                "reason": record.reason,
                "model confidence": round(record.model_confidence, 2),
                "model said": sources_text(record.model_answer),
                "LLM said": sources_text(record.llm_answer),
                "cost ($)": round(record.cost_spent, 5),
                "time (ms)": round(record.latency_ms),
            }
        )
    return rows
