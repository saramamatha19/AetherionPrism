"""Fine-tune Laya on our labels (an experiment, Phase 5b): does it beat model B once it learns our policy?

Fair, like every other model: Laya learns from the FIT slice only; its thresholds come from the
threshold slice; eval is only for the final score. Everything it makes lives in finetuned/ (not in git).

Run:  uv run python -m prism.laya_finetune dry                 (check the data: no training)
      uv run python -m prism.laya_finetune train <epochs>      (→ finetuned/laya_ft_vN; ~15 min/epoch)
      uv run python -m prism.laya_finetune eval <version> …    (e.g. laya_ft_v1 laya_ft_v2)
      uv run python -m prism.laya_finetune system <version>    (full system: Laya + gate + gpt-4o)
"""

import json
import sys
import time
from pathlib import Path

from prism.laya_sources import SOURCE_QUESTIONS, laya_questions
from prism.system_eval import full_system_numbers

LAYA_DATA_DIR = Path("finetuned/laya_data")
FINETUNED_DIR = Path(
    "finetuned"
)  # each fine-tuned Laya: finetuned/laya_ft_v1, v2, … (never overwritten)


def write_laya_training_file(questions, path: Path) -> None:
    """One JSONL line per question: Laya's 7 yes/no questions + the right answers from our labels.

    "expected" is True for every source in the label and False for the others,
    so multi-label (two sources) and no-search (all False) are both taught.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for q in questions:
            expected = {}
            for source in SOURCE_QUESTIONS:
                expected[source] = source in q.labels
            row = {"state": q.text, "questions": laya_questions(), "expected": expected}
            file.write(json.dumps(row) + "\n")


def finetuned_scores(agent, text: str) -> dict:
    """Ask a FINE-TUNED Laya about one question → {"jira": 0.91, ...} (probability of "yes")."""
    result = agent.system_one(text, laya_questions())
    scores = {}
    for source in SOURCE_QUESTIONS:
        scores[source] = result["answers"][source]["noul"]
    return scores


def record_finetuned_scores(agent, version: str, questions, question_set: str) -> dict:
    """Score every question once with a fine-tuned Laya and save it (resume, like the LLM)."""
    path = Path("recordings") / f"{version}_scores_{question_set}.jsonl"
    saved = {}
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            saved[record["id"]] = record["scores"]
    with open(path, "a") as file:
        for q in questions:
            if q.id in saved:
                continue
            saved[q.id] = finetuned_scores(agent, q.text)
            file.write(json.dumps({"id": q.id, "scores": saved[q.id]}) + "\n")
            file.flush()
    return saved


if __name__ == "__main__":
    from laya.train import TrainConfig, dry_run, finetune

    from prism.config import load_settings
    from prism.data import load_questions, split_train

    settings = load_settings()
    fit, _threshold, _gate = split_train(
        load_questions("data/train.csv"),
        settings.threshold_slice_size,
        settings.gate_slice_size,
        settings.seed,
    )
    fit_file = LAYA_DATA_DIR / "fit.jsonl"
    write_laya_training_file(fit, fit_file)
    print(f"Training file: {fit_file} ({len(fit)} questions × {len(SOURCE_QUESTIONS)} yes/no each)")

    command = ""
    if len(sys.argv) >= 2:
        command = sys.argv[1]

    if command == "dry":
        stats = dry_run(str(fit_file), "english", TrainConfig(epochs=1, seed=settings.seed))
        print(json.dumps(stats, indent=2, default=str)[:2000])

    elif command == "train" and len(sys.argv) == 3:
        from prism.registry import next_version

        # Epochs are a choice → versions are compared on the THRESHOLD slice, never on eval.
        # The rest of Laya's settings are its own published defaults.
        epochs = int(sys.argv[2])
        out_dir = FINETUNED_DIR / next_version("laya_ft", FINETUNED_DIR)  # never overwrites
        print(f"Fine-tuning Laya for {epochs} epoch(s) → {out_dir}")
        summary = finetune(
            str(fit_file), "english", str(out_dir), TrainConfig(epochs=epochs, seed=settings.seed)
        )
        print(json.dumps(summary, indent=2, default=str)[:3000])

    elif command == "eval" and len(sys.argv) >= 3:
        import laya

        from prism.compare import percentile
        from prism.evaluate import evaluate
        from prism.laya_sources import load_laya_scores, top_one
        from prism.llm_report import make_llm_student
        from prism.pipeline import make_student
        from prism.registry import load_model
        from prism.thresholds import apply_thresholds, choose_thresholds

        train = load_questions("data/train.csv")
        _fit, threshold_slice, _gate = split_train(
            train, settings.threshold_slice_size, settings.gate_slice_size, settings.seed
        )
        eval_questions = load_questions("data/eval.csv")

        # Score the threshold slice and eval once per fine-tuned version (saved; resumes)
        versions = sys.argv[2:]
        finetuned = {}  # version → (threshold-slice scores, eval scores, latencies)
        for version in versions:
            agent = laya.load(str(FINETUNED_DIR / version))
            finetuned_scores(agent, "warm-up")  # the first call builds the model; not timed
            print(f"Scoring the threshold slice and eval with {version}…")
            threshold_scores = record_finetuned_scores(agent, version, threshold_slice, "threshold")
            eval_scores = record_finetuned_scores(agent, version, eval_questions, "eval")
            latencies = []
            for q in eval_questions[:200]:
                start = time.perf_counter()
                finetuned_scores(agent, q.text)
                latencies.append((time.perf_counter() - start) * 1000)
            finetuned[version] = (threshold_scores, eval_scores, latencies)
            del agent  # free memory before loading the next one

        def laya_students(scores_threshold, scores_eval):
            """Thresholds and the top-1 cut chosen on the threshold slice → 2 students,
            plus the top-1 exact on the threshold slice (used to CHOOSE between versions)."""
            all_scores = []
            truths = []
            for q in threshold_slice:
                all_scores.append(scores_threshold[q.id])
                truths.append(q.labels)
            thresholds = choose_thresholds(all_scores, truths)

            best_cut = 0.0
            best_right = -1
            for i in range(0, 100):
                cut = i / 100
                right = 0
                for q in threshold_slice:
                    right += top_one(scores_threshold[q.id], cut) == q.labels
                if right > best_right:
                    best_cut = cut
                    best_right = right

            by_threshold = {}
            by_top = {}
            for q in eval_questions:
                by_threshold[q.text] = apply_thresholds(scores_eval[q.id], thresholds)
                by_top[q.text] = top_one(scores_eval[q.id], best_cut)

            def threshold_student(text):
                return by_threshold[text]

            def top_student(text):
                return by_top[text]

            return threshold_student, top_student, best_right / len(threshold_slice)

        zs_threshold, zs_top, _ = laya_students(
            load_laya_scores(threshold_slice, "threshold"), load_laya_scores(eval_questions, "eval")
        )
        students = [
            ("Laya zero-shot (thresholds)", zs_threshold),
            ("Laya zero-shot (top-1)", zs_top),
        ]
        notes = []
        for version, (threshold_scores, eval_scores, latencies) in finetuned.items():
            report = json.loads((FINETUNED_DIR / version / "train_report.json").read_text())
            epochs_used = report.get("training", {}).get("epochs", "?")
            ft_threshold, ft_top, threshold_slice_top1 = laya_students(
                threshold_scores, eval_scores
            )
            label = f"Laya FINE-TUNED {version}, {epochs_used} epoch(s)"
            students.append((f"{label} (thresholds)", ft_threshold))
            students.append((f"{label} (top-1)", ft_top))
            notes.append(
                f"{version}: {epochs_used} epoch(s) │ threshold-slice top-1 exact "
                f"{threshold_slice_top1:.1%} (used to CHOOSE between versions) │ "
                f"speed p50 {percentile(latencies, 50):.0f} ms, p95 {percentile(latencies, 95):.0f} ms"
            )

        model, model_thresholds, _gate_value = load_model("tfidf_ftv2_bge_logreg_v2")
        students.append(
            ("model B (tfidf_ftv2_bge_logreg_v2)", make_student(model, model_thresholds))
        )
        students.append(
            (
                f"gpt-4o prompt {settings.llm_prompt_version}",
                make_llm_student(eval_questions, settings),
            )
        )

        lines = [
            "# Why not Laya? Laya (zero-shot and fine-tuned) vs our models (eval, 1,200 questions)",
            "",
            "Laya = a ready-made decision model (ModernBERT-large, 421M), asked 7 yes/no questions (one per",
            "source). Fine-tuned = trained on the fit slice (the same data model B learned from). All of",
            "Laya's thresholds and top-1 cuts were chosen on the threshold slice (train), never on eval.",
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
        lines.append(
            "| FULL SYSTEM (model B + gpt-4o v4, Step 5.6) | 70.0% | 84.2% | 90.7% | 52.1% | 73.2% |"
        )
        lines += ["", "Fine-tuned versions (model B: ~7 ms; target < 50 ms; Laya cost $0):"]
        for note in notes:
            lines.append(f"- {note}")

        table = "\n".join(lines)
        print(table)
        Path("results/laya_comparison.md").write_text(table + "\n")
        print("\nSaved to results/laya_comparison.md")

    elif command == "system" and len(sys.argv) == 3:
        import laya

        from prism.confidence import confidence
        from prism.cost import llm_call_cost
        from prism.gate import choose_gate, risk_coverage
        from prism.llm_answer_eval import answers_file, load_llm_answers, read_answers
        from prism.logreg import scores as model_b_scores
        from prism.pipeline import routing_confidences
        from prism.registry import load_card, load_model
        from prism.thresholds import apply_thresholds, choose_thresholds

        version = sys.argv[2]
        train = load_questions("data/train.csv")
        _fit, threshold_slice, gate_slice = split_train(
            train, settings.threshold_slice_size, settings.gate_slice_size, settings.seed
        )
        eval_questions = load_questions("data/eval.csv")

        agent = laya.load(str(FINETUNED_DIR / version))
        finetuned_scores(agent, "warm-up")
        print(f"Scoring the threshold, gate and eval questions with {version} (saved; resumes)…")
        laya_threshold = record_finetuned_scores(agent, version, threshold_slice, "threshold")
        laya_gate = record_finetuned_scores(agent, version, gate_slice, "gate")
        laya_eval = record_finetuned_scores(agent, version, eval_questions, "eval")

        # Laya's thresholds: on the threshold slice (the same function as model B)
        all_scores = []
        truths = []
        for q in threshold_slice:
            all_scores.append(laya_threshold[q.id])
            truths.append(q.labels)
        laya_thresholds = choose_thresholds(all_scores, truths)

        def answers_and_confidences(scores_by_id, thresholds, questions):
            answers = {}
            confidences = {}
            for q in questions:
                answers[q.id] = apply_thresholds(scores_by_id[q.id], thresholds)
                confidences[q.id], _ = confidence(scores_by_id[q.id], thresholds)
            return answers, confidences

        def gate_for(gate_answers, gate_confidences, rule_on):
            """The gate: chosen on the GATE slice with the same rule as model B (accuracy_floor)."""
            confidence_list = []
            correct = []
            for q in gate_slice:
                confidence_list.append(gate_confidences[q.id])
                correct.append(gate_answers[q.id] == q.labels)
            routing = routing_confidences(confidence_list, gate_slice, rule_on)
            return choose_gate(risk_coverage(routing, correct), settings.accuracy_floor)

        llm_answers = load_llm_answers(eval_questions, settings)
        llm_records = read_answers(answers_file(settings, "eval"))
        llm_costs = {}
        for q in eval_questions:
            r = llm_records[q.id]
            llm_costs[q.id] = llm_call_cost(
                r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"], settings
            )

        rows = []
        # Laya, without and with the negation rule
        gate_answers, gate_confidences = answers_and_confidences(
            laya_gate, laya_thresholds, gate_slice
        )
        eval_answers, eval_confidences = answers_and_confidences(
            laya_eval, laya_thresholds, eval_questions
        )
        for rule_on in [False, True]:
            gate = gate_for(gate_answers, gate_confidences, rule_on)
            numbers = full_system_numbers(
                eval_questions,
                eval_answers,
                eval_confidences,
                llm_answers,
                llm_costs,
                gate,
                rule_on,
            )
            rule_text = "+ negation rule" if rule_on else "no rule"
            rows.append((f"Laya {version} + gpt-4o ({rule_text}), gate {gate}", numbers))

        # Model B, exactly the same way (its saved gate and rule)
        model_b, model_b_thresholds, model_b_gate = load_model("tfidf_ftv2_bge_logreg_v2")
        model_b_rule = load_card("tfidf_ftv2_bge_logreg_v2")["settings"].get(
            "negation_to_llm", False
        )
        model_b_scores_by_id = {}
        for q in eval_questions:
            model_b_scores_by_id[q.id] = model_b_scores(model_b, q.text)
        b_answers, b_confidences = answers_and_confidences(
            model_b_scores_by_id, model_b_thresholds, eval_questions
        )
        numbers = full_system_numbers(
            eval_questions,
            b_answers,
            b_confidences,
            llm_answers,
            llm_costs,
            model_b_gate,
            model_b_rule,
        )
        rows.append((f"model B + gpt-4o (+ negation rule), gate {model_b_gate}", numbers))

        lines = [
            "# Full system: Laya vs model B as the small model (eval, 1,200 questions)",
            "",
            "Both: answer when sure (gate chosen on the gate slice, 80% rule), gpt-4o prompt v4 otherwise.",
            "",
            "| full system | all exact | real-world | company | hard | all needed | small model answers "
            "| hard: confident mistakes / 100 | cost / 1M |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, n in rows:
            lines.append(
                f"| {name} | {n['all_exact']:.1%} | {n['real_world']:.1%} | {n['company']:.1%} "
                f"| {n['hard']:.1%} | {n['all_needed']:.1%} | {n['small_model_share']:.0%} "
                f"| {n['hard_confident_mistakes_per_100']:.0f} | ${n['cost_per_million']:,.0f} |"
            )
        table = "\n".join(lines)
        print(table)
        Path("results/laya_full_system.md").write_text(table + "\n")
        print("\nSaved to results/laya_full_system.md")

    else:
        raise SystemExit(
            "Usage: uv run python -m prism.laya_finetune dry | train <epochs> | eval <version> … "
            "| system <version>"
        )
