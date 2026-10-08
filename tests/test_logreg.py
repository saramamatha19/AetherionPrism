"""Tests for the judges: train on a tiny dataset and check they learned the obvious."""

from prism.data import Question
from prism.logreg import scores, train
from prism.tfidf import build_tfidf


def q(text, *labels):
    return Question(id=text, text=text, labels=frozenset(labels), question_type="", part="")


TINY = [
    q("is the bug fixed", "jira"),
    q("open a ticket for the bug", "jira"),
    q("what did people say in chat", "slack"),
    q("check the slack channel", "slack"),
    q("did the client reply to my email", "gmail"),
    q("search my inbox", "gmail"),
]


def test_judges_learn_the_obvious():
    model = train(TINY, features=build_tfidf())
    s = scores(model, "is the ticket fixed")
    assert s["jira"] > s["slack"] and s["jira"] > s["gmail"]
    assert set(s) == {"kb", "jira", "confluence", "slack", "notion", "gmail", "web"}
