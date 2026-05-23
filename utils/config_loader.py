"""Configuration loader — reads and validates config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Loads, validates, and provides access to the agent configuration.

    The configuration is loaded from a YAML file and exposed as a nested
    dictionary.  Validation ensures that required keys are present and that
    values fall within acceptable ranges.

    Usage::

        config = ConfigLoader.load("config.yaml")
        model = config["llm"]["model"]
    """

    _REQUIRED_KEYS: dict[str, list[str]] = {
        "llm": ["base_url", "model", "temperature", "max_tokens", "top_p"],
        "agent": ["name", "loop_interval_seconds", "max_iterations",
                  "observation_window_size", "working_memory_size",
                  "confidence_threshold"],
        "memory": ["episodic_memory_file", "max_episodic_entries"],
        "environment": ["type", "scenario"],
        "logging": ["level", "file"],
    }

    @classmethod
    def load(cls, config_path: str | Path = "config.yaml") -> dict[str, Any]:
        """Load and validate the configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            A validated configuration dictionary.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If required keys are missing or values are invalid.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        if config is None:
            raise ValueError("Configuration file is empty.")

        cls._validate(config)
        cls._resolve_paths(config, config_path)
        return config

    @classmethod
    def _validate(cls, config: dict[str, Any]) -> None:
        """Ensure all required keys are present and values are plausible."""
        for section, keys in cls._REQUIRED_KEYS.items():
            if section not in config:
                raise ValueError(f"Missing required config section: '{section}'")
            for key in keys:
                if key not in config[section]:
                    raise ValueError(
                        f"Missing required key '{key}' in section '{section}'"
                    )

        # Value range checks
        llm = config["llm"]
        if not (0.0 <= llm["temperature"] <= 2.0):
            raise ValueError("llm.temperature must be between 0.0 and 2.0")
        if llm["max_tokens"] < 1:
            raise ValueError("llm.max_tokens must be >= 1")
        if not (0.0 < llm["top_p"] <= 1.0):
            raise ValueError("llm.top_p must be between 0.0 (exclusive) and 1.0")

        agent = config["agent"]
        if agent["loop_interval_seconds"] < 0.1:
            raise ValueError("agent.loop_interval_seconds must be >= 0.1")
        if agent["max_iterations"] < 1:
            raise ValueError("agent.max_iterations must be >= 1")
        if not (0.0 < agent["confidence_threshold"] < 1.0):
            raise ValueError(
                "agent.confidence_threshold must be between 0.0 and 1.0 (exclusive)"
            )

        memory = config["memory"]
        if memory["max_episodic_entries"] < 1:
            raise ValueError("memory.max_episodic_entries must be >= 1")

    @classmethod
    def _resolve_paths(cls, config: dict[str, Any], config_path: Path) -> None:
        """Resolve relative paths in the config relative to the config file location."""
        base_dir = config_path.parent.resolve()

        # Resolve episodic memory file path
        mem_file = Path(config["memory"]["episodic_memory_file"])
        if not mem_file.is_absolute():
            config["memory"]["episodic_memory_file"] = str(base_dir / mem_file)

        # Resolve log file path
        log_file = Path(config["logging"]["file"])
        if not log_file.is_absolute():
            config["logging"]["file"] = str(base_dir / log_file)

        # Ensure directories exist
        for file_key in (
            config["memory"]["episodic_memory_file"],
            config["logging"]["file"],
        ):
            Path(file_key).parent.mkdir(parents=True, exist_ok=True)
