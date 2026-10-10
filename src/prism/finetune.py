"""Fine-tune an embedding model on our labels (Phase 4c): make pairs, then train on them."""

import json
import random
import tempfile
from datetime import datetime
from pathlib import Path

from prism.baselines import keyword_rules
from prism.registry import data_fingerprint, next_version

FINETUNED_DIR = Path("finetuned")


def overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """How much two sets of sources overlap, from 0 (nothing shared) to 1 (identical)."""
    if not a and not b:  # both "no search" → the same answer
        return 1.0
    return len(a & b) / len(a | b)


def make_pairs(questions, seed: int) -> tuple[list[tuple[str, str, float]], dict]:
    """For each question: one "same" pair, one "different" pair, and tricky pairs for negation."""
    rng = random.Random(seed)

    # Group questions by their exact set of sources, e.g. {jira} → [q1, q7, …]
    groups = {}
    for q in questions:
        if q.labels not in groups:
            groups[q.labels] = []
        groups[q.labels].append(q)

    pairs = []
    counts = {"same": 0, "different": 0, "tricky": 0}

    for q in questions:
        # 1. "same": another question with exactly the same sources
        others = []
        for o in groups[q.labels]:
            if o.id != q.id:
                others.append(o)
        if others:
            o = rng.choice(others)
            pairs.append((q.text, o.text, overlap(q.labels, o.labels)))
            counts["same"] += 1

        # 2. "different": a random question with different sources
        o = rng.choice(questions)
        while o.labels == q.labels:
            o = rng.choice(questions)
        pairs.append((q.text, o.text, overlap(q.labels, o.labels)))
        counts["different"] += 1

        # 3. "tricky": the question mentions a tool's clue word, but doesn't need that tool
        mentioned_but_not_needed = keyword_rules(q.text) - q.labels
        for source in sorted(mentioned_but_not_needed):
            needs_it = []
            for o in questions:
                if source in o.labels:
                    needs_it.append(o)
            o = rng.choice(needs_it)
            pairs.append((q.text, o.text, overlap(q.labels, o.labels)))
            counts["tricky"] += 1

    return pairs, counts


# Phrases that say "don't search X", written from the labelling rules (rule 2: negation removes)
NEGATIONS = {
    "jira": ["not the ticket", "skip jira", "not in jira"],
    "slack": ["not slack", "not the chat"],
    "gmail": ["not the email", "not my inbox"],
    "confluence": ["not the design doc", "not confluence"],
    "notion": ["not my notes", "not notion"],
    "kb": ["not the faq", "not the help article"],
    "web": ["not google", "not the web"],
}


def augment_negations(questions, count: int, seed: int) -> list[tuple]:
    """Make `count` new questions: "<not X>, <original>", where X is a source it does NOT need.
    Returns (new text, original text, labels, excluded source). The labels don't change."""
    rng = random.Random(seed)

    with_sources = []
    for q in questions:
        if q.labels:  # skip "no search" questions (small talk)
            with_sources.append(q)

    augmented = []
    for i in range(count):
        q = rng.choice(with_sources)

        not_needed = []
        for source in NEGATIONS:
            if source not in q.labels:
                not_needed.append(source)
        excluded = rng.choice(not_needed)

        phrase = rng.choice(NEGATIONS[excluded])
        new_text = phrase + ", " + q.text[0].lower() + q.text[1:]
        augmented.append((new_text, q.text, q.labels, excluded))

    return augmented


def augmented_pairs(augmented, questions, seed: int) -> list[tuple[str, str, float]]:
    """2 pairs per augmented question: close to its original, far from a question needing X."""
    rng = random.Random(seed)
    pairs = []
    for new_text, original_text, labels, excluded in augmented:
        # 1. "not X, <question>" means the same as "<question>"
        pairs.append((new_text, original_text, 1.0))

        # 2. ...and is NOT a question about X
        needs_it = []
        for o in questions:
            if excluded in o.labels:
                needs_it.append(o)
        o = rng.choice(needs_it)
        pairs.append((new_text, o.text, overlap(labels, o.labels)))
    return pairs


def finetune(base_model_name, pairs, epochs, batch_size, seed, out_dir: Path) -> None:
    """Nudge the embedding model so that how close each pair is ≈ its target. Save it to out_dir."""
    if out_dir.exists():
        raise ValueError(f"{out_dir} already exists: a fine-tuned model is never overwritten")

    # Imported here so the rest of the project doesn't need the training libraries
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )

    texts_a = []
    texts_b = []
    targets = []
    for a, b, target in pairs:
        texts_a.append(a)
        texts_b.append(b)
        targets.append(target)
    data = Dataset.from_dict({"sentence1": texts_a, "sentence2": texts_b, "score": targets})

    model = SentenceTransformer(base_model_name)

    with tempfile.TemporaryDirectory() as scratch:  # trainer's rough notebook, deleted afterwards
        args = SentenceTransformerTrainingArguments(
            output_dir=scratch,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            seed=seed,
            save_strategy="no",  # save only the final model, not in-between copies
            logging_steps=50,  # write down the loss every 50 batches
            report_to="none",  # don't send training logs to any website
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=data,
            loss=losses.CosineSimilarityLoss(model),
        )
        trainer.train()

        # The learning curve: the loss every 50 batches (it should go down)
        log = []
        for entry in trainer.state.log_history:
            if "loss" in entry:
                log.append(
                    {
                        "step": entry["step"],
                        "epoch": entry["epoch"],
                        "loss": entry["loss"],
                    }
                )

    model.save(str(out_dir))
    (out_dir / "training").mkdir()
    (out_dir / "training" / "log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    from prism.config import load_settings
    from prism.data import load_questions, split_fit_in_half, split_train
    from prism.pipeline import BGE

    settings = load_settings()
    fit, _, _ = split_train(
        load_questions("data/train.csv"),
        settings.threshold_slice_size,
        settings.gate_slice_size,
        settings.seed,
    )
    # Which questions BGE learns from:
    #   split fix ON  → only the finetune half (the judges later train on the OTHER half)
    #   split fix OFF → the whole fit slice (BGE and judges learn from the same questions)
    if settings.finetune_on_fit_half:
        finetune_questions, _judges_half = split_fit_in_half(fit, settings.seed)
        trained_on = "fit slice, finetune half only (split fix)"
    else:
        finetune_questions = fit
        trained_on = "fit slice"

    pairs, counts = make_pairs(finetune_questions, settings.seed)
    augmented = augment_negations(
        finetune_questions, settings.finetune_augment_count, settings.seed
    )
    extra = augmented_pairs(augmented, finetune_questions, settings.seed)
    pairs = pairs + extra
    counts["augmented"] = len(extra)
    print(f"Fine-tuning on {len(finetune_questions)} questions ({trained_on})")
    print(f"Pairs: {counts} → {len(pairs)} in total")

    version = next_version("bge_ft", FINETUNED_DIR)
    out_dir = FINETUNED_DIR / version
    finetune(
        BGE,
        pairs,
        settings.finetune_epochs,
        settings.finetune_batch_size,
        settings.seed,
        out_dir,
    )

    card = {
        "version": version,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "base_model": BGE,
        "trained_on": trained_on,
        "questions_used": len(finetune_questions),
        "epochs": settings.finetune_epochs,
        "batch_size": settings.finetune_batch_size,
        "augment_count": settings.finetune_augment_count,
        "seed": settings.seed,
        "pairs": counts,
        "train_data_fingerprint": data_fingerprint("data/train.csv"),
    }
    (out_dir / "card.json").write_text(json.dumps(card, indent=2))
    print(f"\nSaved as {out_dir}/")
