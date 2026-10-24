"""Tests for Router 2 (GET /prediction/{question_id}). The test app (client) is in conftest.py."""

from test_intakeRouter import one


def test_predicted_question(client):
    client.post("/intake", json={"questions": [one("108", "look at my email", ["gmail"])]})
    reply = client.get("/prediction/108")
    assert reply.status_code == 200
    answer = reply.json()
    assert answer["status"] == "predicted"
    assert answer["prism_answer"] == ["gmail"]
    assert answer["model_version"] == client.app.state.router.model_version
    assert "question" not in answer and "intent_answer" not in answer  # only Prism's side goes out


def test_waiting_question(client):
    client.app.state.store.save_question("109", "leave policy?", ["kb"])  # saved, not predicted
    answer = client.get("/prediction/109").json()
    assert answer["status"] == "waiting"
    assert answer["prism_answer"] is None


def test_unknown_id(client):
    assert client.get("/prediction/999").status_code == 404
