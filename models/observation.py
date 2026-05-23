"""Observation data model — represents a single environmental observation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Observation:
    """A structured observation from the environment.

    Attributes:
        timestamp: When the observation was made.
        source: Origin of the observation (e.g. 'environment', 'feedback', 'system').
        content: The main textual content of the observation.
        relevance_score: Computed relevance score (0.0–1.0) used for attention filtering.
        observation_id: Unique identifier for this observation.
        metadata: Additional key-value pairs providing context.
    """

    timestamp: datetime
    source: str
    content: str
    relevance_score: float = 0.0
    observation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize the observation to a dictionary."""
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Observation:
        """Deserialize an observation from a dictionary."""
        data_copy = data.copy()
        if isinstance(data_copy.get("timestamp"), str):
            data_copy["timestamp"] = datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)

    def __str__(self) -> str:
        return f"[{self.source}] {self.content} (rel={self.relevance_score:.2f})"
