"""Save and load trained models: models/<recipe>_v<N>/. A saved model is never overwritten."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib

MODELS_DIR = Path("models")


def data_fingerprint(path) -> str:
    """A short code computed from the file's contents. Any change to the file → a different code."""
    contents = Path(path).read_bytes()
    return hashlib.sha256(contents).hexdigest()[:12]


def next_version(recipe: str, models_dir: Path = MODELS_DIR) -> str:
    """tfidf_logreg_v1, then v2, v3, ... (one more than the highest that already exists)."""
    prefix = recipe + "_v"
    highest = 0
    if models_dir.exists():
        for folder in models_dir.iterdir():
            if folder.name.startswith(prefix):
                number = folder.name[len(prefix) :]
                if number.isdigit() and int(number) > highest:
                    highest = int(number)
    return prefix + str(highest + 1)


def save_model(
    model,
    thresholds,
    gate,
    recipe,
    settings,
    eval_result,
    gate_slice_result,
    eval_gate_result,
    models_dir: Path = MODELS_DIR,
    extra_card: dict | None = None,
) -> str:
    """Save the model + its ID card into a new version folder. Returns the version name.
    extra_card: more notes for the card (e.g. which reviewed batches it learned from)."""
    version = next_version(recipe, models_dir)
    folder = models_dir / version
    folder.mkdir(parents=True)  # error if the folder already exists → never overwrite

    joblib.dump(
        {"model": model, "thresholds": thresholds, "gate": gate},
        folder / "model.joblib",
    )

    card = {
        "version": version,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "recipe": recipe,
        "settings": settings.model_dump(),
        "train_data_fingerprint": data_fingerprint("data/train.csv"),
        "thresholds": thresholds,
        "gate": gate,
        "gate_slice": {
            "coverage": gate_slice_result["coverage"],
            "accuracy": gate_slice_result["accuracy"],
        },
        "eval": {
            "all_exact": eval_result["all"]["exact_match"],
            "real_world_exact": eval_result["by_part"]["real-world"]["exact_match"],
            "all_needed": eval_result["all"]["found_all_needed"],
            "coverage": eval_gate_result["coverage"],
            "kept_accuracy": eval_gate_result["accuracy"],
        },
    }
    if extra_card:
        card.update(extra_card)
    (folder / "card.json").write_text(json.dumps(card, indent=2))
    return version


def load_model(version: str, models_dir: Path = MODELS_DIR):
    """Load a saved model by its version name → (model, thresholds, gate). Only load your own files."""
    saved = joblib.load(models_dir / version / "model.joblib")
    if "gate" not in saved:
        raise ValueError(
            f"{version} has no gate (saved before Phase 4). Train a new version: "
            "uv run python -m prism.pipeline"
        )
    return saved["model"], saved["thresholds"], saved["gate"]


def load_card(version: str, models_dir: Path = MODELS_DIR) -> dict:
    """Read a saved model's ID card (card.json): settings, thresholds, gate, eval scores."""
    return json.loads((models_dir / version / "card.json").read_text())


# ---------- Which model is live (the "champion"), and the history of every change ----------

CHAMPION_FILE = "champion.json"  # inside models/: which version is live, explained in plain words
PROMOTIONS_FILE = "promotions.jsonl"  # inside models/: every change, appended, never rewritten


def read_champion(models_dir: Path = MODELS_DIR) -> dict:
    """The live model's pointer file. Missing → ERROR (the router never guesses a model)."""
    path = models_dir / CHAMPION_FILE
    if not path.exists():
        raise ValueError(
            f"No {path}: promote a model first (uv run python -m prism.retrain promote …)"
        )
    return json.loads(path.read_text())


def promote(
    version: str,
    by: str,
    what_it_is: str,
    why: str,
    evidence: dict,
    details: str,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """Make `version` the live model. Writes champion.json + one history line. Always explained."""
    if not (models_dir / version).exists():
        raise ValueError(f"No saved model {version} in {models_dir}")
    if not why.strip() or not what_it_is.strip():
        raise ValueError("A promotion must say what the model is and why (in plain words)")

    previous_version = None
    if (models_dir / CHAMPION_FILE).exists():
        previous_version = read_champion(models_dir)["version"]

    today = datetime.now().strftime("%Y-%m-%d")
    champion = {
        "version": version,
        "since": today,
        "promoted_by": by,
        "what_it_is": what_it_is,
        "why": why,
        "evidence_on_eval": evidence,
        "details": details,
    }
    (models_dir / CHAMPION_FILE).write_text(json.dumps(champion, indent=2) + "\n")

    history_line = {
        "when": today,
        "by": by,
        "from": previous_version,
        "to": version,
        "why": why,
        "evidence_on_eval": evidence,
    }
    with open(models_dir / PROMOTIONS_FILE, "a") as file:  # append: the history is never rewritten
        file.write(json.dumps(history_line) + "\n")
    return champion


def rollback(by: str, why: str, models_dir: Path = MODELS_DIR) -> dict:
    """Go back to the champion BEFORE the current one (recorded as a promotion too)."""
    history = []
    for line in (models_dir / PROMOTIONS_FILE).read_text().splitlines():
        history.append(json.loads(line))
    last = history[-1]
    if last["from"] is None:
        raise ValueError("Nothing to roll back to: this is the first champion")

    old_card = json.loads((models_dir / last["from"] / "card.json").read_text())
    return promote(
        last["from"],
        by,
        what_it_is=f"Rolled back to {last['from']} (the model live before {last['to']})",
        why=why,
        evidence=old_card["eval"],  # its own saved evidence
        details=f"rollback from {last['to']}",
        models_dir=models_dir,
    )
