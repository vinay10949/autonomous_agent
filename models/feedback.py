"""Feedback data model — captures the outcome of an executed action."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FeedbackSignal(str, Enum):
    """Signal types for action feedback."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Feedback:
    """Feedback from the environment after an action is executed.

    Attributes:
        action_id: The ID of the action this feedback corresponds to.
        success: Whether the action completed successfully.
        signal: Qualitative signal — positive, negative, or neutral.
        message: Human-readable description of the outcome.
        new_observations: Any new information uncovered by the action.
        feedback_id: Unique identifier for this feedback entry.
        timestamp: When the feedback was generated.
    """

    action_id: str
    success: bool
    signal: str
    message: str
    new_observations: list[str] = field(default_factory=list)
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Serialize the feedback to a dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "action_id": self.action_id,
            "success": self.success,
            "signal": self.signal,
            "message": self.message,
            "new_observations": self.new_observations,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Feedback:
        """Deserialize feedback from a dictionary."""
        data_copy = data.copy()
        if isinstance(data_copy.get("timestamp"), str):
            data_copy["timestamp"] = datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return f"Feedback({status}): {self.message}"
