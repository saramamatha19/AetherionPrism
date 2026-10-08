"""The 7 yes/no judges (logistic regression), one per source. Works on ANY features."""

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from prism.data import SOURCES, Question


@dataclass
class LogRegModel:
    """A trained model: the features (words → numbers) + one judge per source."""

    features: object  # e.g. build_tfidf(), after learning
    judges: dict[str, LogisticRegression | None]  # "jira" → the jira judge; None = never needed


def train(questions: list[Question], features) -> LogRegModel:
    """Learn the features and the 7 judges from the fit slice."""
    numbers = features.fit_transform([q.text for q in questions])
    judges = {}
    for source in SOURCES:
        needed = [source in q.labels for q in questions]  # True/False per question
        if not any(needed):  # never needed here → nothing to learn → always says 0
            judges[source] = None
            continue
        judge = LogisticRegression(max_iter=1000)
        judge.fit(numbers, needed)
        judges[source] = judge
    return LogRegModel(features=features, judges=judges)


def scores(model: LogRegModel, text: str) -> dict[str, float]:
    """Ask all 7 judges about one question → {"jira": 0.91, "slack": 0.12, ...}."""
    numbers = model.features.transform([text])
    return {
        source: 0.0 if judge is None else float(judge.predict_proba(numbers)[0, 1])
        for source, judge in model.judges.items()
    }


def top_words(model: LogRegModel, source: str, n: int = 5) -> list[str]:
    """The n words/chunks that push this judge most towards "yes" (for debugging)."""
    if model.judges[source] is None:
        return []
    columns = model.features.get_feature_names_out()
    weights = model.judges[source].coef_[0]
    best = weights.argsort()[::-1][:n]
    return [columns[i] for i in best]
