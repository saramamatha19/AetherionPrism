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
) -> str:
    """Save the model + its ID card into a new version folder. Returns the version name."""
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
