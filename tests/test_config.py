"""Tests for the settings checker: good settings load, bad ones are rejected."""

import pytest
from pydantic import ValidationError

from prism.config import load_settings


def write_settings(tmp_path, text):
    """Write a small settings file in a throwaway folder and return its path."""
    path = tmp_path / "settings.yaml"
    path.write_text(text)
    return path


# 1. The real file works
def test_real_settings_file_loads():
    settings = load_settings("configs/default.yaml")
    assert settings.mode in ("model_only", "gated", "shadow")


# 2. Forgot a setting → error (the prototype bug)
def test_missing_setting_is_an_error(tmp_path):
    path = write_settings(tmp_path, "mode: shadow\n")
    with pytest.raises(ValidationError, match="accuracy_floor"):
        load_settings(path)


# 3. Silly accuracy floor (15%) → error
def test_silly_accuracy_floor_is_an_error(tmp_path):
    path = write_settings(tmp_path, "mode: shadow\naccuracy_floor: 0.15\n")
    with pytest.raises(ValidationError, match="accuracy_floor"):
        load_settings(path)


# 4. Typo in a setting name → error
def test_typo_is_an_error(tmp_path):
    path = write_settings(tmp_path, "mode: shadow\naccuracy_floor: 0.8\ngat: 0.4\n")
    with pytest.raises(ValidationError, match="gat"):
        load_settings(path)


# 5. Mode that isn't one of the 3 allowed words → error
def test_wrong_mode_is_an_error(tmp_path):
    path = write_settings(tmp_path, "mode: shado\naccuracy_floor: 0.8\n")
    with pytest.raises(ValidationError, match="mode"):
        load_settings(path)
