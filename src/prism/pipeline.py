"""Put the small model together: split → train judges → thresholds → gate → student."""

import sys

from sklearn.pipeline import FeatureUnion

from prism.confidence import confidence
from prism.data import load_questions, split_fit_in_half, split_train
from prism.embeddings import SentenceEmbeddings
from prism.gate import choose_gate, coverage_at, risk_coverage
from prism.logreg import scores, train
from prism.ruleNegation import has_negation
from prism.tfidf import build_tfidf
from prism.thresholds import apply_thresholds, choose_thresholds

BGE = "BAAI/bge-small-en-v1.5"
# Fine-tuned BGE models, made by `uv run python -m prism.finetune` (in finetuned/, not in git)
FINETUNED_BGE = "finetuned/bge_ft_v1"  # v1: fine-tuned on the whole fit slice
FINETUNED_BGE_V2 = "finetuned/bge_ft_v2"  # v2: + 1,000 augmented negation questions
FINETUNED_BGE_SPLIT = "finetuned/bge_ft_v3"  # v3: split fix, fine-tuned on the finetune half only

RECIPE_NAMES = [
    "tfidf",
    "bge",
    "tfidf_bge",
    "ft_bge",
    "tfidf_ft_bge",
    "tfidf_ftv2_bge",
    "tfidf_ftsplit_bge",
]

# Recipes whose fine-tuned BGE studied only the finetune half of fit.
# Their judges must train on the OTHER half (judges_half), never on questions BGE has seen.
SPLIT_FIX_RECIPES = ["tfidf_ftsplit_bge"]


def build_features(recipe: str):
    """The "words → numbers" part of each recipe. A new embedding model later = one more `if`."""
    if recipe == "tfidf":
        return build_tfidf()
    if recipe == "bge":
        return SentenceEmbeddings(BGE)
    if recipe == "tfidf_bge":
        return FeatureUnion([("tfidf", build_tfidf()), ("emb", SentenceEmbeddings(BGE))])
    if recipe == "ft_bge":
        return SentenceEmbeddings(FINETUNED_BGE)
    if recipe == "tfidf_ft_bge":
        return FeatureUnion([("tfidf", build_tfidf()), ("emb", SentenceEmbeddings(FINETUNED_BGE))])
    if recipe == "tfidf_ftv2_bge":
        return FeatureUnion(
            [("tfidf", build_tfidf()), ("emb", SentenceEmbeddings(FINETUNED_BGE_V2))]
        )
    if recipe == "tfidf_ftsplit_bge":
        return FeatureUnion(
            [("tfidf", build_tfidf()), ("emb", SentenceEmbeddings(FINETUNED_BGE_SPLIT))]
        )
    raise ValueError(f"Unknown recipe '{recipe}'. Choose one of: {', '.join(RECIPE_NAMES)}")


def train_small_model(
    settings, features, judges_on_judges_half: bool = False, extra_fit_questions=None
):
    """Judges on the fit slice → thresholds on the threshold slice → gate on the gate slice.

    judges_on_judges_half=True (split fix): the judges train only on the judges half of fit,
    the questions the fine-tuned BGE never studied, so they don't become over-confident.
    extra_fit_questions (learning loop): reviewed real questions the judges ALSO learn from.
    The threshold and gate slices never change, so every version is measured with the same ruler.
    """
    all_train = load_questions("data/train.csv")
    fit, threshold, gate_slice = split_train(
        all_train,
        settings.threshold_slice_size,
        settings.gate_slice_size,
        settings.seed,
    )

    # Which questions the 7 judges learn from (split fix → only the half BGE never saw)
    if judges_on_judges_half:
        _finetune_half, judges_questions = split_fit_in_half(fit, settings.seed)
    else:
        judges_questions = fit

    if extra_fit_questions:  # the learning loop: + reviewed real questions
        judges_questions = judges_questions + list(extra_fit_questions)

    model = train(judges_questions, features=features)

    all_scores = []
    truths = []
    for q in threshold:
        all_scores.append(scores(model, q.text))
        truths.append(q.labels)
    thresholds = choose_thresholds(all_scores, truths)

    confidences, correct = confidences_and_correct(model, thresholds, gate_slice)
    # The gate is chosen on what it will really see: with the negation rule (if on) applied
    routing = routing_confidences(confidences, gate_slice, settings.negation_to_llm)
    gate = choose_gate(risk_coverage(routing, correct), settings.accuracy_floor)
    gate_slice_result = coverage_at(routing, correct, gate)

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


# A routing number below any gate → the model never keeps the question → the LLM answers
SENT_TO_LLM_BY_RULE = -1.0


def routing_confidences(confidences, questions, negation_to_llm: bool):
    """The numbers the GATE looks at: the real confidences, except that questions caught by
    the negation rule get SENT_TO_LLM_BY_RULE (always the LLM).

    The real confidences list is NOT changed, so the model's true confidence is kept for logs.
    """
    routing = []
    for i in range(len(questions)):
        if negation_to_llm and has_negation(questions[i].text):
            routing.append(SENT_TO_LLM_BY_RULE)
        else:
            routing.append(confidences[i])
    return routing


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

    if len(sys.argv) != 2 or sys.argv[1] not in RECIPE_NAMES:
        raise SystemExit(
            "Usage: uv run python -m prism.pipeline <recipe>   "
            f"(recipes: {', '.join(RECIPE_NAMES)})"
        )
    recipe = sys.argv[1]

    settings = load_settings()
    model, thresholds, gate, gate_slice_result = train_small_model(
        settings,
        features=build_features(recipe),
        judges_on_judges_half=(recipe in SPLIT_FIX_RECIPES),
    )

    eval_questions = load_questions("data/eval.csv")
    result = evaluate(make_student(model, thresholds), eval_questions)
    print_report(recipe + " + logreg", result)

    # Does the gate keep its promise on the final exam?
    confidences, correct = confidences_and_correct(model, thresholds, eval_questions)
    routing = routing_confidences(confidences, eval_questions, settings.negation_to_llm)
    eval_gate_result = coverage_at(routing, correct, gate)
    print(f"\nGate {gate} (accuracy_floor {settings.accuracy_floor})")
    for name, r in [("gate slice", gate_slice_result), ("eval", eval_gate_result)]:
        print(
            f"  {name:<11} model keeps {r['coverage']:.1%}, right on {r['accuracy']:.1%} of those"
        )

    version = save_model(
        model,
        thresholds,
        gate,
        recipe + "_logreg",
        settings,
        result,
        gate_slice_result,
        eval_gate_result,
    )
    print(f"\nSaved as models/{version}/")
