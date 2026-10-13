"""Read and check the settings file (configs/default.yaml)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelPrice(BaseModel):
    """One LLM's prices, $ per 1 million tokens, and where they came from."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    cached_input: float = Field(ge=0)
    output: float = Field(ge=0)
    source: str  # where and when the prices were checked


class Settings(BaseModel):
    """Every setting the project uses. No defaults: a missing setting is an error."""

    model_config = ConfigDict(extra="forbid")  # unknown setting (typo) → error

    mode: Literal["model_only", "gated", "shadow"]  # only these 3 words allowed
    accuracy_floor: float = Field(gt=0.5, lt=1)  # must be between 0.5 and 1
    seed: int
    threshold_slice_size: int = Field(gt=0)  # must be a positive number
    gate_slice_size: int = Field(gt=0)
    finetune_epochs: int = Field(ge=1)  # at least 1 pass
    finetune_batch_size: int = Field(ge=1)
    finetune_augment_count: int = Field(ge=0)  # 0 = no augmentation
    finetune_on_fit_half: bool  # true = split fix: fine-tune on the finetune half of fit only
    negation_to_llm: bool  # true = negation questions always go to the LLM
    llm_model: str  # an exact model version, e.g. "gpt-4o-2024-11-20"
    llm_timeout_seconds: float = Field(gt=0)
    llm_max_retries: int = Field(ge=0)
    llm_prompt_version: str  # e.g. "v1"; the prompts themselves live in llm.py
    llm_prices: dict[str, ModelPrice]  # model name → its prices
    prompt_practice_per_type: int = Field(ge=1)
    audit_rate: float = Field(ge=0, le=1)  # share of "model sure" questions spot-checked by the LLM
    spot_check_rate: float = Field(
        ge=0, le=1
    )  # shadow: share of confident agreements a human checks
    go_live_accuracy: float = Field(gt=0.5, lt=1)  # the model's accuracy needed to go live
    go_live_min_questions: int = Field(ge=1)  # minimum shadow questions before deciding
    retrain_min_new_rows: int = Field(ge=1)  # new reviewed rows needed before retraining
    batch_max_questions: int = Field(ge=1)  # most questions routed in one batch


def load_settings(path: str | Path = "configs/default.yaml") -> Settings:
    """Read the YAML file and check it. Raises a clear error if anything is wrong."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Settings(**data)
