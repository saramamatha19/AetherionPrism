"""Tests for the intake export: prism.db → a decision log the existing shadow tools can read."""

import pytest

from prism.intakeExport import export, translate
from prism.intakeStore import IntakeStore
from prism.shadowReport import load_run, shadow_report

SAME_NAMES = {
    "kb": "kb",
    "jira": "jira",
    "confluence": "confluence",
    "slack": "slack",
    "notion": "notion",
    "gmail": "gmail",
    "web": "web",
}


@pytest.fixture
def store(tmp_path):
    store = IntakeStore(tmp_path / "prism.db")
    store.save_question("1", "status of ABC-12?", ["jira"])
    store.save_prediction("1", ["jira"], True, 0.9, "model_v1")  # sure, agrees
    store.save_question("2", "leave policy?", ["kb"])
    store.save_prediction("2", ["web"], False, 0.2, "model_v1")  # unsure, disagrees
    store.save_question("3", "is it on sharepoint?", ["sharepoint"])
    store.save_prediction("3", ["kb"], True, 0.8, "model_v1")  # a name Prism doesn't know
    store.save_question("4", "what did Ravi say?", ["slack"])  # waiting: not predicted
    return store


def test_translate():
    assert translate(["jira", "kb"], SAME_NAMES) == ["jira", "kb"]
    assert translate(["knowledge_base"], {"knowledge_base": "kb"}) == ["kb"]
    assert translate([], SAME_NAMES) == []  # no source
    assert translate(["jira", "sharepoint"], SAME_NAMES) is None


def test_export_counts(store, tmp_path):
    summary = export(store, SAME_NAMES, tmp_path / "run")
    assert summary["exported"] == 2  # 1 and 2
    assert summary["waiting"] == 1  # 4
    assert summary["skipped_unknown"] == 1 and summary["unknown_names"] == {"sharepoint": 1}


def test_existing_shadow_tools_read_it(store, tmp_path):
    export(store, SAME_NAMES, tmp_path / "run")
    decisions = load_run(tmp_path / "run")  # the real shadow tools
    assert len(decisions) == 2
    report = shadow_report(decisions, spot_check_rate=0.0, seed=42)
    assert report["agreement_all"] == 0.5  # 1 agrees, 2 doesn't
    assert report["confident"] == 1  # only 1 was sure
    assert decisions[1]["would_escalate"] is True  # 2 was unsure


def test_platform_names_translated(tmp_path):
    store = IntakeStore(tmp_path / "prism.db")
    store.save_question("1", "leave policy?", ["knowledge_base"])
    store.save_prediction("1", ["kb"], True, 0.9, "model_v1")
    export(store, {"knowledge_base": "kb"}, tmp_path / "run")
    assert load_run(tmp_path / "run")[0]["llm_answer"] == ["kb"]


def test_since_only_newer(store, tmp_path):
    store.run("UPDATE questions SET received_at = '2026-11-01T09:00:00' WHERE question_id = '1'")
    store.run("UPDATE questions SET received_at = '2026-11-12T09:00:00' WHERE question_id = '2'")
    summary = export(store, SAME_NAMES, tmp_path / "run", since="2026-11-10")
    assert summary["exported"] == 1  # only 2 (3 is unknown, also newer: still skipped)


def test_wrong_table_is_refused(store, tmp_path):
    with pytest.raises(ValueError, match="isn't one of"):
        export(store, {"jira": "jiraa"}, tmp_path / "run")


def test_never_overwrites(store, tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        export(store, SAME_NAMES, tmp_path / "run")


def test_nothing_to_export_makes_no_folder(tmp_path):
    summary = export(IntakeStore(tmp_path / "prism.db"), SAME_NAMES, tmp_path / "run")
    assert summary["exported"] == 0 and not (tmp_path / "run").exists()
