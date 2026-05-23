"""Structured JSON-line logger for agent trace data."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Custom formatter that emits one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        # Merge any extra fields passed via the `extra` kwarg
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry, default=str)


class AgentLogger:
    """Dual-output logger: human-readable console + structured JSONL file.

    The console handler uses Rich-compatible formatting for readability,
    while the file handler writes one JSON object per line for downstream
    processing and analysis.

    Usage::

        logger = AgentLogger("AutonomousAgent", log_file="./logs/agent_trace.jsonl")
        logger.info("Agent started")
        logger.log_decision(decision_dict)
    """

    def __init__(
        self,
        name: str = "AutonomousAgent",
        log_file: str | Path | None = None,
        level: str = "INFO",
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.propagate = False

        # Avoid duplicate handlers on re-initialization
        if not self._logger.handlers:
            # Console handler — plain text for Rich to colorize
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._logger.level)
            console_fmt = logging.Formatter(
                "%(asctime)s │ %(levelname)-8s │ %(message)s",
                datefmt="%H:%M:%S",
            )
            console_handler.setFormatter(console_fmt)
            self._logger.addHandler(console_handler)

            # File handler — JSON lines
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(
                    log_path, encoding="utf-8", mode="a"
                )
                file_handler.setLevel(self._logger.level)
                file_handler.setFormatter(_JsonFormatter())
                self._logger.addHandler(file_handler)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(msg, extra={"extra_data": kwargs} if kwargs else {})

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(msg, extra={"extra_data": kwargs} if kwargs else {})

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(msg, extra={"extra_data": kwargs} if kwargs else {})

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(msg, extra={"extra_data": kwargs} if kwargs else {})

    def log_decision(
        self,
        decision: dict,
        include_reasoning: bool = True,
    ) -> None:
        """Log a cognition-layer decision with its full reasoning trace.

        Args:
            decision: A dictionary representation of the Decision model.
            include_reasoning: Whether to include the full reasoning text.
        """
        entry = {
            "event": "decision",
            "decision_id": decision.get("decision_id", "unknown"),
            "selected_action": decision.get("selected_action", "unknown"),
            "confidence": decision.get("confidence", 0.0),
        }
        if include_reasoning:
            entry["reasoning"] = decision.get("reasoning", "")
            entry["candidate_actions"] = decision.get("candidate_actions", [])
        self._logger.info(
            f"Decision: {entry['selected_action']} (conf={entry['confidence']:.2f})",
            extra={"extra_data": entry},
        )

    def log_action(self, action: dict, feedback: dict | None = None) -> None:
        """Log an action and its feedback."""
        entry = {
            "event": "action",
            "action_id": action.get("action_id", "unknown"),
            "action_type": action.get("action_type", "unknown"),
            "params": action.get("params", {}),
        }
        if feedback:
            entry["feedback"] = feedback
        self._logger.info(
            f"Action: {entry['action_type']} → "
            f"{'OK' if feedback and feedback.get('success') else 'PENDING' if not feedback else 'FAIL'}",
            extra={"extra_data": entry},
        )
