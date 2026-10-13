"""Tests for the decision log (in a throwaway folder, never the real logs/)."""

import csv
from datetime import date

from test_decisionRecord import example_record

from prism.decisionLog import DecisionLog, read_decisions


def todays_files(log):
    return log.files_for(date.today().isoformat())


def test_one_line_per_decision_in_both_files(tmp_path):
    log = DecisionLog(logs_dir=tmp_path)
    log.write(example_record())
    log.write(example_record(question="look at my email"))
    json_path, csv_path = todays_files(log)
    assert len(json_path.read_text().splitlines()) == 2
    rows = list(csv.reader(csv_path.open()))
    assert len(rows) == 3  # the header row + 2 decisions
    assert rows[0][0] == "time"  # the header is first


def test_json_reads_back_exactly(tmp_path):
    log = DecisionLog(logs_dir=tmp_path)
    log.write(example_record())
    json_path, _ = todays_files(log)
    back = read_decisions(json_path)[0]
    assert back["model_answer"] == ["jira", "slack"]
    assert back["llm_answer"] is None and back["would_escalate"] is False


def test_question_with_comma_stays_one_csv_cell(tmp_path):
    log = DecisionLog(logs_dir=tmp_path)
    log.write(example_record(question="check jira, then slack"))
    _, csv_path = todays_files(log)
    rows = list(csv.reader(csv_path.open()))
    assert rows[1][1] == "check jira, then slack"  # one cell, not split at the comma


def test_count_shadow_questions(tmp_path):
    import json

    from prism.decisionLog import count_shadow_questions

    with open(tmp_path / "decisions_2026-10-13.jsonl", "w") as file:
        for mode in ["shadow", "gated", "shadow"]:
            file.write(json.dumps({"mode": mode}) + "\n")
    assert count_shadow_questions(tmp_path) == 2  # the gated one isn't counted
