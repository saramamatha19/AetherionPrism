"""Put the small model together: split → train judges → choose thresholds → student."""

from prism.data import load_questions, split_train
from prism.logreg import scores, train
from prism.thresholds import apply_thresholds, choose_thresholds


def train_small_model(settings, features):
    """Train the judges on the fit slice, then choose thresholds on the threshold slice."""
    all_train = load_questions("data/train.csv")
    fit, threshold, gate = split_train(
        all_train, settings.threshold_slice_size, settings.gate_slice_size, settings.seed
    )

    model = train(fit, features=features)

    all_scores = []
    truths = []
    for q in threshold:
        all_scores.append(scores(model, q.text))
        truths.append(q.labels)
    thresholds = choose_thresholds(all_scores, truths)

    return model, thresholds


def make_student(model, thresholds):
    """Wrap model + thresholds as a "student": question text → set of sources."""

    def student(text):
        return apply_thresholds(scores(model, text), thresholds)

    return student


if __name__ == "__main__":
    from prism.config import load_settings
    from prism.data import load_questions
    from prism.evaluate import evaluate, print_report
    from prism.registry import save_model
    from prism.tfidf import build_tfidf

    settings = load_settings()
    model, thresholds = train_small_model(settings, features=build_tfidf())
    result = evaluate(make_student(model, thresholds), load_questions("data/eval.csv"))
    print_report("tfidf + logreg", result)

    version = save_model(model, thresholds, "tfidf_logreg", settings, result)
    print(f"\nSaved as models/{version}/")
