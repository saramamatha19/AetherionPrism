"""Test that the student wrapper turns scores into a set of sources."""

from prism.data import SOURCES, Question
from prism.logreg import train
from prism.pipeline import make_student
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


def test_student_returns_sources_that_pass():
    model = train(TINY, features=build_tfidf())
    thresholds = {}
    for source in SOURCES:
        thresholds[source] = 0.3
    student = make_student(model, thresholds)
    assert "jira" in student("is the ticket fixed")
