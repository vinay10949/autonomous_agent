"""Working Memory — short-term rolling buffer for recent observations and context."""

from __future__ import annotations

from collections import deque
from typing import Any

from models.observation import Observation


class WorkingMemory:
    """A fixed-capacity rolling buffer that holds the most recent observations.

    Working memory acts as the agent's short-term memory, maintaining a
    sliding window of the latest observations.  Older entries are
    automatically evicted when the buffer reaches capacity, ensuring that
    the cognition layer always operates on recent, relevant context.

    The buffer is implemented as a deque for O(1) append and O(1) eviction
    at the left end.

    Attributes:
        capacity: Maximum number of items the buffer can hold.
        _buffer: Internal deque storing Observation objects.
    """

    def __init__(self, capacity: int = 20) -> None:
        """Initialize working memory with the given capacity.

        Args:
            capacity: Maximum number of observations to retain.
        """
        self.capacity = max(1, capacity)
        self._buffer: deque[Observation] = deque(maxlen=self.capacity)
        self._context: dict[str, Any] = {}

    def add(self, observation: Observation) -> None:
        """Add an observation to working memory.

        If the buffer is at capacity, the oldest observation is dropped.

        Args:
            observation: The Observation to add.
        """
        self._buffer.append(observation)

    def get_recent(self, n: int | None = None) -> list[Observation]:
        """Return the N most recent observations (newest first).

        Args:
            n: Number of recent observations to return. If None, return all.

        Returns:
            A list of Observation objects, newest first.
        """
        items = list(self._buffer)
        items.reverse()
        if n is not None:
            return items[:n]
        return items

    def get_all(self) -> list[Observation]:
        """Return all observations in insertion order (oldest first).

        Returns:
            A list of all Observation objects currently in working memory.
        """
        return list(self._buffer)

    def clear(self) -> None:
        """Remove all observations from working memory."""
        self._buffer.clear()

    def set_context(self, key: str, value: Any) -> None:
        """Store a key-value pair in the working memory context.

        The context dictionary is separate from the observation buffer
        and is used for agent state that doesn't fit the observation model,
        such as the current task, agent mode, or environment flags.

        Args:
            key: Context key.
            value: Context value.
        """
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the working memory context.

        Args:
            key: Context key.
            default: Default value if key is not found.

        Returns:
            The stored value or the default.
        """
        return self._context.get(key, default)

    @property
    def size(self) -> int:
        """Current number of observations in the buffer."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached its capacity."""
        return len(self._buffer) >= self.capacity

    def summary(self) -> dict[str, Any]:
        """Return a summary of working memory state for logging/debugging."""
        return {
            "size": self.size,
            "capacity": self.capacity,
            "is_full": self.is_full,
            "context_keys": list(self._context.keys()),
            "latest_source": self._buffer[-1].source if self._buffer else None,
        }

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"WorkingMemory(size={self.size}/{self.capacity})"
