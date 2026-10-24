"""Tests for Router 1 (POST /intake), with FastAPI's test client: no server, no real cron, no LLM.
The test app (fixtures app_client and client) lives in conftest.py, shared with Router 2's tests."""

from prism.intakeRouter import predict_unpredicted
from prism.intakeStore import IntakeStore


def one(question_id, question, intent_answer):
    return {"question_id": question_id, "question": question, "intent_answer": intent_answer}


def test_saves_and_predicts(client):
    reply = client.post("/intake", json={"questions": [one("108", "look at my email", ["gmail"])]})
    assert reply.status_code == 200
    assert reply.json() == {"saved": 1, "skipped_duplicates": 0, "skipped_empty": 0, "predicted": 1}

    found = client.app.state.store.get_prediction("108")
    assert found["intent_answer"] == ["gmail"]
    assert found["prism_answer"] == ["gmail"]
    assert found["model_version"] == client.app.state.router.model_version
    assert isinstance(found["model_sure"], bool)


def test_same_question_sent_twice(client):
    body = {"questions": [one("108", "look at my email", ["gmail"])]}
    client.post("/intake", json=body)
    reply = client.post("/intake", json=body).json()  # the cron's retry
    assert reply["saved"] == 0 and reply["skipped_duplicates"] == 1


def test_unknown_platform_name_is_kept(client):
    body = {"questions": [one("109", "is it on sharepoint?", ["sharepoint"])]}
    assert client.post("/intake", json=body).json()["saved"] == 1
    assert client.app.state.store.get_prediction("109")["intent_answer"] == ["sharepoint"]


def test_empty_question_is_skipped_and_counted(client):
    body = {"questions": [one("110", "   ", ["kb"]), one("111", "leave policy?", ["kb"])]}
    reply = client.post("/intake", json=body).json()
    assert reply["saved"] == 1 and reply["skipped_empty"] == 1


def test_long_question_saved_in_full(client):
    long_question = (
        "where is the onboarding checklist for new joiners? " * 400
    )  # ~20,000 characters
    reply = client.post("/intake", json={"questions": [one("112", long_question, ["kb"])]}).json()
    assert reply["predicted"] == 1
    assert client.app.state.store.get_prediction("112")["question"] == long_question.strip()


def test_missing_id_is_refused(client):
    reply = client.post("/intake", json={"questions": [one("", "leave policy?", ["kb"])]})
    assert reply.status_code == 422  # refused by the IntakeQuestion shape


def test_too_many_in_one_send_is_refused(client):
    limit = client.app.state.router.settings.intake_max_questions
    questions = []
    for number in range(limit + 1):
        questions.append(one(str(number), "leave policy?", ["kb"]))
    reply = client.post("/intake", json={"questions": questions})
    assert reply.status_code == 400
    assert client.app.state.store.get_unpredicted() == []  # nothing saved


class BrokenRouter:
    """A model that always fails, to test that one failure never loses a question."""

    def route(self, question_text, write_log=True, mode=None):
        raise RuntimeError("model broke")


def test_failed_prediction_is_retried_later(client, tmp_path):
    store = IntakeStore(tmp_path / "retry.db")
    store.save_question("113", "look at my email", ["gmail"])

    assert predict_unpredicted(store, BrokenRouter()) == 0
    assert len(store.get_unpredicted()) == 1  # still saved, waiting

    assert predict_unpredicted(store, client.app.state.router) == 1  # the next send fixes it
    assert store.get_unpredicted() == []
