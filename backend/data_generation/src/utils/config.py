"""Configuration management for the STT data generation pipeline."""

import os
from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = [
    "youtube",
    "quality_validation",
    "stt",
    "vad",
    "synthesis",
    "output_dirs",
    "reproducibility",
]


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Supports environment variable overrides for leaf values using the format
    ``DG_<SECTION>_<KEY>`` (all uppercase). For example, the environment
    variable ``DG_STT_DEVICE`` overrides ``config["stt"]["device"]``.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed and validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required top-level keys are missing.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(config)}")

    _validate_required_keys(config)
    _apply_env_overrides(config)

    return config


def _validate_required_keys(config: dict[str, Any]) -> None:
    """Check that all required top-level keys are present.

    Args:
        config: Parsed configuration dictionary.

    Raises:
        ValueError: If any required key is missing.
    """
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Override config leaf values with matching environment variables.

    Convention: ``DG_<SECTION>_<KEY>`` maps to ``config[section][key]``.
    Type coercion matches the existing config value type (int, float, bool, str).

    Args:
        config: Configuration dictionary to mutate in-place.
    """
    prefix = "DG_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        parts = env_key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue

        section, key = parts
        if section not in config or not isinstance(config[section], dict):
            continue
        if key not in config[section]:
            continue

        original = config[section][key]
        config[section][key] = _coerce(env_val, original)


def _coerce(value: str, reference: Any) -> Any:
    """Cast *value* to the same type as *reference*.

    Args:
        value: String value from an environment variable.
        reference: Existing config value used to infer target type.

    Returns:
        *value* cast to the type of *reference*, or the original string if
        casting is not possible.
    """
    if isinstance(reference, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(reference, int):
        try:
            return int(value)
        except ValueError:
            return value
    if isinstance(reference, float):
        try:
            return float(value)
        except ValueError:
            return value
    return value
