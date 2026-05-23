"""Episodic Memory — persistent storage of past decision episodes.

Each episode records the full cycle: observation → decision → action → feedback.
Episodes are persisted as JSONL (one JSON object per line) for durability
across agent restarts and for post-hoc analysis.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class EpisodicMemory:
    """Persistent, append-only store for decision episodes.

    An episode is a tuple of (observation, decision, action, feedback) that
    records one complete PCA cycle.  Episodes are written as JSONL lines
    and can be retrieved for reflection, similarity matching, or audit.

    When the number of stored episodes exceeds ``max_entries``, the oldest
    entries are pruned to keep the file within bounds.

    Attributes:
        file_path: Path to the JSONL storage file.
        max_entries: Maximum number of episodes to retain.
    """

    def __init__(self, file_path: str | Path, max_entries: int = 100) -> None:
        """Initialize episodic memory.

        Args:
            file_path: Path to the JSONL file for persistent storage.
            max_entries: Maximum number of episodes to keep.
        """
        self.file_path = Path(file_path)
        self.max_entries = max(1, max_entries)
        self._episodes: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load existing episodes from the JSONL file."""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            return

        self._episodes = []
        with open(self.file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        self._episodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def _save(self) -> None:
        """Persist all episodes to the JSONL file (full rewrite)."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as fh:
            for episode in self._episodes:
                fh.write(json.dumps(episode, default=str) + "\n")

    def store(
        self,
        observation: dict | None = None,
        decision: dict | None = None,
        action: dict | None = None,
        feedback: dict | None = None,
        reflection: dict | None = None,
    ) -> str:
        """Store a new episode and persist it.

        Args:
            observation: Serialized Observation dict.
            decision: Serialized Decision dict.
            action: Serialized Action dict.
            feedback: Serialized Feedback dict.
            reflection: Serialized reflection dict from the cognition layer.

        Returns:
            The episode ID.
        """
        from uuid import uuid4

        episode_id = str(uuid4())[:8]
        episode: dict[str, Any] = {
            "episode_id": episode_id,
            "timestamp": datetime.now().isoformat(),
            "observation": observation or {},
            "decision": decision or {},
            "action": action or {},
            "feedback": feedback or {},
            "reflection": reflection or {},
        }

        self._episodes.append(episode)

        # Append to file immediately for durability
        with open(self.file_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(episode, default=str) + "\n")

        # Prune if needed
        if len(self._episodes) > self.max_entries:
            self._episodes = self._episodes[-self.max_entries :]
            self._save()  # Full rewrite to remove pruned entries

        return episode_id

    def retrieve_recent(self, n: int = 5) -> list[dict[str, Any]]:
        """Retrieve the N most recent episodes.

        Args:
            n: Number of episodes to retrieve.

        Returns:
            A list of episode dicts, newest first.
        """
        return list(reversed(self._episodes[-n:]))

    def retrieve_by_similarity(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve episodes similar to the query using simple keyword overlap.

        This is a lightweight similarity measure based on shared keywords
        between the query and episode content.  For production use, this
        could be replaced with embedding-based retrieval.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of episode dicts sorted by similarity (highest first).
        """
        query_words = set(query.lower().split())

        scored: list[tuple[float, dict[str, Any]]] = []
        for ep in self._episodes:
            # Combine all text fields for matching
            text_parts = []
            for field in ("observation", "decision", "action", "feedback", "reflection"):
                val = ep.get(field, {})
                if isinstance(val, dict):
                    text_parts.extend(str(v).lower() for v in val.values())
                else:
                    text_parts.append(str(val).lower())

            episode_words = set(" ".join(text_parts).split())
            overlap = len(query_words & episode_words)
            score = overlap / max(len(query_words), 1)
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    @property
    def size(self) -> int:
        """Current number of stored episodes."""
        return len(self._episodes)

    def clear(self) -> None:
        """Remove all episodes from memory and delete the storage file."""
        self._episodes = []
        if self.file_path.exists():
            self.file_path.write_text("")

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"EpisodicMemory(size={self.size}/{self.max_entries})"
