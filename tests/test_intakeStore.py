"""Tests for the intake store, each with its own empty file in a temporary folder (never the real one)."""

import pytest

from prism.intakeStore import IntakeStore


@pytest.fixture
def store(tmp_path):
    return IntakeStore(tmp_path / "prism.db")


def test_save_and_read_back(store):
    assert store.save_question("101", "where is the leave policy?", ["kb"]) is True
    found = store.get_prediction("101")
    assert found["question"] == "where is the leave policy?"
    assert found["intent_answer"] == ["kb"]
    assert found["prism_answer"] is None  # the model hasn't run yet


def test_same_id_twice_is_skipped(store):
    store.save_question("101", "where is the leave policy?", ["kb"])
    assert store.save_question("101", "a different text", ["jira"]) is False
    assert store.get_prediction("101")["question"] == "where is the leave policy?"  # first one kept


def test_unpredicted_until_prism_answers(store):
    store.save_question("101", "where is the leave policy?", ["kb"])
    store.save_question("102", "status of ticket ABC-12?", ["jira"])
    assert len(store.get_unpredicted()) == 2

    assert store.save_prediction("101", ["kb"], True, 0.93, "model_v1") is True
    unpredicted = store.get_unpredicted()
    assert len(unpredicted) == 1 and unpredicted[0]["question_id"] == "102"

    found = store.get_prediction("101")
    assert found["prism_answer"] == ["kb"] and found["model_version"] == "model_v1"
    assert found["model_sure"] is True and found["confidence"] == 0.93


def test_unknown_id(store):
    assert store.get_prediction("999") is None
    assert store.save_prediction("999", ["kb"], True, 0.93, "model_v1") is False


def test_platform_names_kept_as_sent(store):
    store.save_question("104", "is it on sharepoint?", ["sharepoint"])  # a name Prism doesn't know
    assert store.get_prediction("104")["intent_answer"] == ["sharepoint"]


def test_quotes_in_a_question_are_safe(store):
    question = "what's Ravi's \"design\" status; DROP TABLE questions"
    store.save_question("103", question, ["slack"])
    assert store.get_prediction("103")["question"] == question


def test_data_survives_a_restart(tmp_path):
    IntakeStore(tmp_path / "prism.db").save_question("101", "where is the leave policy?", ["kb"])
    reopened = IntakeStore(tmp_path / "prism.db")  # like a new container on the same volume
    assert reopened.get_prediction("101")["intent_answer"] == ["kb"]
