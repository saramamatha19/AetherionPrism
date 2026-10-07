"""Read and check the settings file (configs/default.yaml)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Every setting the project uses. No defaults: a missing setting is an error."""

    model_config = ConfigDict(extra="forbid")  # unknown setting (typo) → error

    mode: Literal["model_only", "gated", "shadow"]  # only these 3 words allowed
    accuracy_floor: float = Field(gt=0.5, lt=1)  # must be between 0.5 and 1


def load_settings(path: str | Path = "configs/default.yaml") -> Settings:
    """Read the YAML file and check it. Raises a clear error if anything is wrong."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Settings(**data)
