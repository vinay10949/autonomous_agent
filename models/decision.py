"""Decision data model — captures the agent's reasoning and choice."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Decision:
    """A decision made by the cognition layer, including the full reasoning trace.

    Attributes:
        timestamp: When the decision was made.
        situation_summary: A concise description of the assessed situation.
        candidate_actions: List of candidate actions with their scores, each as a dict
            with keys like 'name', 'description', 'effectiveness', 'risk',
            'information_gain', 'alignment', 'total_score'.
        selected_action: The name of the action that was selected.
        reasoning: The chain-of-thought reasoning that led to the decision.
        confidence: Overall confidence in the decision (0.0–1.0).
        decision_id: Unique identifier for this decision.
    """

    timestamp: datetime
    situation_summary: str
    candidate_actions: list[dict] = field(default_factory=list)
    selected_action: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        """Serialize the decision to a dictionary."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "situation_summary": self.situation_summary,
            "candidate_actions": self.candidate_actions,
            "selected_action": self.selected_action,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Decision:
        """Deserialize a decision from a dictionary."""
        data_copy = data.copy()
        if isinstance(data_copy.get("timestamp"), str):
            data_copy["timestamp"] = datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)

    def __str__(self) -> str:
        return (
            f"Decision({self.decision_id}): {self.selected_action} "
            f"(confidence={self.confidence:.2f})"
        )
