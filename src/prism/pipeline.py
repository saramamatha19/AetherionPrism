"""Put the small model together: split → train judges → thresholds → gate → student."""

from prism.confidence import confidence
from prism.data import load_questions, split_train
from prism.gate import choose_gate, coverage_at, risk_coverage
from prism.logreg import scores, train
from prism.thresholds import apply_thresholds, choose_thresholds


def train_small_model(settings, features):
    """Judges on the fit slice → thresholds on the threshold slice → gate on the gate slice."""
    all_train = load_questions("data/train.csv")
    fit, threshold, gate_slice = split_train(
        all_train,
        settings.threshold_slice_size,
        settings.gate_slice_size,
        settings.seed,
    )

    model = train(fit, features=features)

    all_scores = []
    truths = []
    for q in threshold:
        all_scores.append(scores(model, q.text))
        truths.append(q.labels)
    thresholds = choose_thresholds(all_scores, truths)

    confidences, correct = confidences_and_correct(model, thresholds, gate_slice)
    gate = choose_gate(risk_coverage(confidences, correct), settings.accuracy_floor)
    gate_slice_result = coverage_at(confidences, correct, gate)

    return model, thresholds, gate, gate_slice_result


# from gate.py
def confidences_and_correct(model, thresholds, questions):
    """For each question: how sure the model is, and whether its answer was exactly right."""
    confidences = []
    correct = []
    for q in questions:
        s = scores(model, q.text)
        sure, _ = confidence(s, thresholds)
        confidences.append(sure)
        correct.append(apply_thresholds(s, thresholds) == q.labels)
    return confidences, correct


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
    model, thresholds, gate, gate_slice_result = train_small_model(
        settings, features=build_tfidf()
    )

    eval_questions = load_questions("data/eval.csv")
    result = evaluate(make_student(model, thresholds), eval_questions)
    print_report("tfidf + logreg", result)

    # Does the gate keep its promise on the final exam?
    confidences, correct = confidences_and_correct(model, thresholds, eval_questions)
    eval_gate_result = coverage_at(confidences, correct, gate)
    print(f"\nGate {gate} (accuracy_floor {settings.accuracy_floor})")
    for name, r in [("gate slice", gate_slice_result), ("eval", eval_gate_result)]:
        print(
            f"  {name:<11} model keeps {r['coverage']:.1%}, right on {r['accuracy']:.1%} of those"
        )

    version = save_model(
        model,
        thresholds,
        gate,
        "tfidf_logreg",
        settings,
        result,
        gate_slice_result,
        eval_gate_result,
    )
    print(f"\nSaved as models/{version}/")
