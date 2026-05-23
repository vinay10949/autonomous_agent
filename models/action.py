"""Action data model — represents an action the agent intends to execute."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ActionType(str, Enum):
    """Supported action types for the agent."""

    RESPOND = "respond"
    QUERY_ENVIRONMENT = "query_environment"
    WAIT = "wait"
    ESCALATE = "escalate"
    TOOL_USE = "tool_use"


@dataclass
class Action:
    """An action to be executed by the action layer.

    Attributes:
        action_type: The category of action (from ActionType enum).
        params: Parameters for the action, e.g. {'task_id': 'T-001', 'message': '...'}.
        timestamp: When the action was created.
        action_id: Unique identifier for this action.
    """

    action_type: str
    params: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    action_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        """Serialize the action to a dictionary."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "params": self.params,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Action:
        """Deserialize an action from a dictionary."""
        data_copy = data.copy()
        if isinstance(data_copy.get("timestamp"), str):
            data_copy["timestamp"] = datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"Action({self.action_type}): {params_str}"
