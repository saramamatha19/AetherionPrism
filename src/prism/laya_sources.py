"""Laya (a ready-made decision model) asked our 7 source questions (Phase 5b).

Laya answers typed questions about a text in one pass. We use its "noul" type
(the probability the answer is yes): one yes/no question per source, so several sources
can be "yes" (multi-label) or none (no search). The output has the same shape as
model B's scores, so our thresholds, confidence and gate can be reused unchanged.

Run:  uv run python -m prism.laya_sources            (smoke test: 6 example questions)
      uv run python -m prism.laya_sources practice   (Step 5b.2: Laya vs gpt-4o on the practice set)
      uv run python -m prism.laya_sources eval       (Step 5b.3: Laya vs model B vs gpt-4o on eval)
"""

import hashlib
import json
import sys
import time
from pathlib import Path

LAYA_RECORDINGS_DIR = Path("recordings")

# One yes/no question per source, worded from our source descriptions (the same as prompt v4)
SOURCE_QUESTIONS = {
    "kb": "Does answering this need a search in the knowledge base: official help, how-to or policy articles?",
    "jira": "Does answering this need a search in Jira: tickets, bugs, tasks, work status (who owns it, is it fixed, blockers, progress)?",
    "confluence": "Does answering this need a search in Confluence: written team documents such as specs, design docs, runbooks, process docs?",
    "slack": "Does answering this need a search in Slack: chat conversations (what people said, suggested or discussed)?",
    "notion": "Does answering this need a search in Notion: meeting notes, decisions, plans, deadlines, personal or team notes?",
    "gmail": "Does answering this need a search in Gmail: email (what someone wrote, sent or replied)?",
    "web": "Does answering this need a web search: public knowledge outside the company?",
}


def laya_questions() -> dict:
    """Our 7 sources in the shape Laya expects: {id: {"type": "noul", "instructions": ...}}."""
    questions = {}
    for source, instruction in SOURCE_QUESTIONS.items():
        questions[source] = {"type": "noul", "instructions": instruction}
    return questions


def laya_scores(router, text: str) -> dict:
    """Ask Laya about one question → {"jira": 0.91, "slack": 0.12, ...} (probability of "yes")."""
    result = router.predict(text, laya_questions(), model="english")
    scores = {}
    for source in SOURCE_QUESTIONS:
        scores[source] = result["answers"][source]["noul"]
    return scores


def laya_fingerprint() -> str:
    """A short code from our 7 question wordings: change a wording → a new file."""
    text = json.dumps(SOURCE_QUESTIONS, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def laya_scores_file(question_set: str) -> Path:
    """Where Laya's scores for a set of questions are saved."""
    return LAYA_RECORDINGS_DIR / f"laya_scores_{question_set}_{laya_fingerprint()}.jsonl"


def record_laya_scores(questions, question_set: str, router) -> None:
    """Ask Laya every question not saved yet; append one line each (resume, like the LLM)."""
    path = laya_scores_file(question_set)
    path.parent.mkdir(exist_ok=True)
    saved_ids = set()
    if path.exists():
        for line in path.read_text().splitlines():
            saved_ids.add(json.loads(line)["id"])

    with open(path, "a") as file:
        for q in questions:
            if q.id in saved_ids:
                continue
            file.write(json.dumps({"id": q.id, "scores": laya_scores(router, q.text)}) + "\n")
            file.flush()


def load_laya_scores(questions, question_set: str) -> dict:
    """{question id: {source: probability}}. A missing question → ERROR (never silently skipped)."""
    saved = {}
    for line in laya_scores_file(question_set).read_text().splitlines():
        record = json.loads(line)
        saved[record["id"]] = record["scores"]
    scores = {}
    missing = []
    for q in questions:
        if q.id in saved:
            scores[q.id] = saved[q.id]
        else:
            missing.append(q.id)
    if missing:
        raise ValueError(f"{len(missing)} questions have no Laya scores (e.g. {missing[:3]})")
    return scores


def compare_on_practice(router) -> None:
    """Step 5b.2: thresholds on the threshold slice → Laya vs gpt-4o v4 on practice questions
    that are NOT in the threshold slice (both have never been trained on them)."""
    from prism.config import load_settings
    from prism.data import load_questions, prompt_practice_set, split_train
    from prism.evaluate import evaluate, print_table
    from prism.llm_answer_eval import load_llm_answers
    from prism.thresholds import apply_thresholds, choose_thresholds

    settings = load_settings()
    train = load_questions("data/train.csv")
    _fit, threshold_slice, _gate = split_train(
        train, settings.threshold_slice_size, settings.gate_slice_size, settings.seed
    )
    practice = prompt_practice_set(train, settings.prompt_practice_per_type, settings.seed)

    # Test only on practice questions that are NOT used to choose the thresholds
    threshold_ids = set()
    for q in threshold_slice:
        threshold_ids.add(q.id)
    test_questions = []
    for q in practice:
        if q.id not in threshold_ids:
            test_questions.append(q)

    print(f"Laya scores: threshold slice ({len(threshold_slice)}) + practice ({len(practice)})…")
    record_laya_scores(threshold_slice, "threshold", router)
    record_laya_scores(practice, "practice", router)

    # Thresholds: the SAME function as model B (Step 3.4), on the threshold slice
    threshold_scores = load_laya_scores(threshold_slice, "threshold")
    all_scores = []
    truths = []
    for q in threshold_slice:
        all_scores.append(threshold_scores[q.id])
        truths.append(q.labels)
    thresholds = choose_thresholds(all_scores, truths)
    print("Laya thresholds:", thresholds)

    # Laya's answers and gpt-4o v4's recorded answers on the test questions
    practice_scores = load_laya_scores(practice, "practice")
    gpt4o_answers = load_llm_answers(practice, settings, question_set="practice")
    laya_by_text = {}
    gpt4o_by_text = {}
    for q in test_questions:
        laya_by_text[q.text] = apply_thresholds(practice_scores[q.id], thresholds)
        gpt4o_by_text[q.text] = gpt4o_answers[q.id]

    def laya_student(text):
        return laya_by_text[text]

    def gpt4o_student(text):
        return gpt4o_by_text[text]

    print(f"\nTest questions: {len(test_questions)} (practice minus the threshold slice)")
    for name, student in [
        ("Laya (zero-shot)", laya_student),
        (f"gpt-4o {settings.llm_prompt_version}", gpt4o_student),
    ]:
        result = evaluate(student, test_questions)
        print(f"\n===== {name} =====")
        print_table("By question type", {**result["by_type"], "ALL": result["all"]})


def top_one(scores: dict, cut: float) -> frozenset:
    """Keep only Laya's single highest source (or none if even that is below `cut`)."""
    best = None
    for source, probability in scores.items():
        if best is None or probability > scores[best]:
            best = source
    if scores[best] < cut:
        return frozenset()
    return frozenset({best})


def compare_on_eval(router, model_version: str) -> None:
    """Step 5b.3: Laya on all 1,200 eval questions vs model B vs gpt-4o v4 (same questions).
    Every choice for Laya (thresholds, top-1 cut) is made on the threshold slice, never on eval."""
    from prism.compare import percentile
    from prism.config import load_settings
    from prism.data import load_questions, split_train
    from prism.evaluate import evaluate
    from prism.llm_report import make_llm_student
    from prism.pipeline import make_student
    from prism.registry import load_model
    from prism.thresholds import apply_thresholds, choose_thresholds

    settings = load_settings()
    train = load_questions("data/train.csv")
    _fit, threshold_slice, _gate = split_train(
        train, settings.threshold_slice_size, settings.gate_slice_size, settings.seed
    )
    eval_questions = load_questions("data/eval.csv")

    # Laya's scores on eval (saved once) + its speed per question
    print(f"Laya scores on eval ({len(eval_questions)})…")
    record_laya_scores(threshold_slice, "threshold", router)
    record_laya_scores(eval_questions, "eval", router)
    latencies = []
    for q in eval_questions[:200]:  # speed: 200 new questions, one at a time
        start = time.perf_counter()
        laya_scores(router, q.text)
        latencies.append((time.perf_counter() - start) * 1000)

    # Choices made on the threshold slice only
    threshold_scores = load_laya_scores(threshold_slice, "threshold")
    all_scores = []
    truths = []
    for q in threshold_slice:
        all_scores.append(threshold_scores[q.id])
        truths.append(q.labels)
    thresholds = choose_thresholds(all_scores, truths)

    best_cut = 0.0
    best_right = -1
    for i in range(0, 60):
        cut = i / 100
        right = 0
        for q in threshold_slice:
            right += top_one(threshold_scores[q.id], cut) == q.labels
        if right > best_right:
            best_cut = cut
            best_right = right

    eval_scores = load_laya_scores(eval_questions, "eval")
    laya_thresholds_by_text = {}
    laya_top_by_text = {}
    for q in eval_questions:
        laya_thresholds_by_text[q.text] = apply_thresholds(eval_scores[q.id], thresholds)
        laya_top_by_text[q.text] = top_one(eval_scores[q.id], best_cut)

    def laya_thresholds_student(text):
        return laya_thresholds_by_text[text]

    def laya_top_student(text):
        return laya_top_by_text[text]

    model, model_thresholds, _gate_value = load_model(model_version)
    students = [
        ("Laya zero-shot (thresholds)", laya_thresholds_student),
        ("Laya zero-shot (top-1)", laya_top_student),
        (f"model B ({model_version})", make_student(model, model_thresholds)),
        (
            f"gpt-4o prompt {settings.llm_prompt_version}",
            make_llm_student(eval_questions, settings),
        ),
    ]

    lines = [
        "# Why not Laya? Zero-shot Laya vs our models (eval, 1,200 questions)",
        "",
        "Laya = a ready-made decision model (ModernBERT-large, 421M), asked 7 yes/no questions (one per",
        "source). Its thresholds and top-1 cut were chosen on the threshold slice (train), never on eval.",
        "",
        "| system | all exact | real-world | company | hard | all needed |",
        "|---|---|---|---|---|---|",
    ]
    for name, student in students:
        r = evaluate(student, eval_questions)
        bp = r["by_part"]
        lines.append(
            f"| {name} | {r['all']['exact_match']:.1%} | {bp['real-world']['exact_match']:.1%} "
            f"| {bp['company topics']['exact_match']:.1%} | {bp['hard cases']['exact_match']:.1%} "
            f"| {r['all']['found_all_needed']:.1%} |"
        )
    lines += [
        "",
        f"Laya thresholds (chosen on the threshold slice): {thresholds}",
        f"Laya top-1 cut (chosen on the threshold slice): {best_cut}",
        f"Laya speed on this laptop: p50 {percentile(latencies, 50):.0f} ms, "
        f"p95 {percentile(latencies, 95):.0f} ms per question (model B ~7 ms; target < 50 ms). Cost $0.",
    ]
    table = "\n".join(lines)
    print(table)
    Path("results").mkdir(exist_ok=True)
    Path("results/laya_comparison.md").write_text(table + "\n")
    print("\nSaved to results/laya_comparison.md")


if __name__ == "__main__":
    from laya import Router

    print("Loading Laya (downloads ~1.7 GB the first time)…")
    router = Router()
    laya_scores(router, "warm-up")  # the first call builds the model; not timed

    if len(sys.argv) == 2 and sys.argv[1] == "practice":
        compare_on_practice(router)
        raise SystemExit(0)
    if len(sys.argv) == 2 and sys.argv[1] == "eval":
        compare_on_eval(router, "tfidf_ftv2_bge_logreg_v2")
        raise SystemExit(0)

    for text in [
        "is the login bug fixed yet?",
        "what did we decide in mondays meeting about the launch?",
        "not the email, what did people say in slack about the outage?",
        "how do I pay a parking ticket in Boston?",
        "thanks, that was helpful!",
        "look into that thing from earlier",
    ]:
        start = time.perf_counter()
        scores = laya_scores(router, text)
        ms = (time.perf_counter() - start) * 1000
        top = sorted(scores.items(), key=lambda item: -item[1])[:3]
        print(f"{text:<64} {ms:>6.0f} ms   " + "  ".join(f"{s} {p:.2f}" for s, p in top))
