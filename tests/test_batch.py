"""Tests for batches: splitting text, reading files, routing many questions (no real model/LLM)."""

import io

import pandas as pd
import pytest
from test_decisionRecord import example_record

from prism.batch import read_questions_file, results_rows, route_many, split_questions


def test_one_question_per_line():
    text = "is the login bug fixed?\n\n  what is a KV cache  \nthanks"
    assert split_questions(text) == ["is the login bug fixed?", "what is a KV cache", "thanks"]


def test_a_paragraph_is_split_into_questions():
    text = "Is the bug fixed? What did slack say. Check the spec!"
    assert split_questions(text) == ["Is the bug fixed?", "What did slack say.", "Check the spec!"]


def test_dots_inside_words_do_not_split():
    assert split_questions("what changed in v1.2 of the api?") == [
        "what changed in v1.2 of the api?"
    ]


def test_repeats_and_punctuation_only_are_dropped():
    assert split_questions("Bug fixed?\nbug   FIXED?\n???\n...") == ["Bug fixed?"]


def test_txt_file():
    data = b"is the login bug fixed?\nwhat is a KV cache?"
    assert read_questions_file("q.txt", data) == ["is the login bug fixed?", "what is a KV cache?"]


def test_csv_with_a_question_column():
    data = b"id,Question\n1,is the login bug fixed?\n2,\n3,what is a KV cache?\n"
    assert read_questions_file("q.csv", data) == ["is the login bug fixed?", "what is a KV cache?"]


def test_csv_without_a_header_keeps_the_first_row():
    data = b"is the login bug fixed?\nwhat is a KV cache?\n"
    assert read_questions_file("q.csv", data) == ["is the login bug fixed?", "what is a KV cache?"]


def test_excel_file():
    buffer = io.BytesIO()
    pd.DataFrame({"question": ["is the login bug fixed?", "what is a KV cache?"]}).to_excel(
        buffer, index=False
    )
    assert read_questions_file("q.xlsx", buffer.getvalue()) == [
        "is the login bug fixed?",
        "what is a KV cache?",
    ]


def test_other_file_types_are_refused():
    with pytest.raises(ValueError, match="txt, csv, xlsx"):
        read_questions_file("notes.pdf", b"%PDF")


class FakeRouter:
    """Remembers what it was asked; answers with an example record (no model, no LLM)."""

    def __init__(self):
        self.asked = []

    def route(self, question_text, mode=None):
        self.asked.append((question_text, mode))
        return example_record(question=question_text, mode=mode)


def test_route_many_routes_each_question_in_order():
    router = FakeRouter()
    progress = []

    def remember_progress(done, total):
        progress.append((done, total))

    records = route_many(router, ["a?", "b?"], 10, mode="shadow", on_progress=remember_progress)
    assert router.asked == [("a?", "shadow"), ("b?", "shadow")]
    routed_questions = []
    for record in records:
        routed_questions.append(record.question)
    assert routed_questions == ["a?", "b?"]
    assert progress == [(1, 2), (2, 2)]


def test_route_many_refuses_too_many_questions():
    router = FakeRouter()
    with pytest.raises(ValueError, match="limit of 2"):
        route_many(router, ["a?", "b?", "c?"], 2)
    assert router.asked == []  # nothing was routed (no money spent)


def test_results_rows_are_readable():
    row = results_rows([example_record(answer=frozenset())])[0]
    assert row["search in"] == "no search"
    assert row["model said"] == "jira|slack"
    assert row["LLM said"] == ""  # the LLM wasn't asked
