"""Memory Manager — orchestrates working memory and episodic memory operations.

The MemoryManager provides a unified interface for the cognition layer to
store and retrieve information across both short-term (working) and
long-term (episodic) memory systems.  It handles the coordination of
storing decision episodes and providing context for reasoning.
"""

from __future__ import annotations

from typing import Any

from memory.episodic_memory import EpisodicMemory
from memory.working_memory import WorkingMemory
from models.observation import Observation


class MemoryManager:
    """Unified interface over working and episodic memory.

    The manager simplifies the cognitive layer's interactions with memory by
    providing high-level operations such as "record_episode" (which stores an
    entire PCA cycle) and "get_context" (which assembles relevant information
    from both memory systems for the LLM prompt).

    Attributes:
        working: The working memory instance.
        episodic: The episodic memory instance.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the memory manager from the agent configuration.

        Args:
            config: The full agent configuration dictionary.  The 'memory'
                section is used for episodic memory, and the 'agent' section
                for working memory capacity.
        """
        memory_cfg = config.get("memory", {})
        agent_cfg = config.get("agent", {})

        self.working = WorkingMemory(
            capacity=agent_cfg.get("working_memory_size", 20)
        )
        self.episodic = EpisodicMemory(
            file_path=memory_cfg.get("episodic_memory_file", "./data/episodic_memory.json"),
            max_entries=memory_cfg.get("max_episodic_entries", 100),
        )

    def add_observation(self, observation: Observation) -> None:
        """Add an observation to working memory.

        Args:
            observation: The Observation to store.
        """
        self.working.add(observation)

    def record_episode(
        self,
        observation: dict | None = None,
        decision: dict | None = None,
        action: dict | None = None,
        feedback: dict | None = None,
        reflection: dict | None = None,
    ) -> str:
        """Record a complete PCA cycle as an episode.

        Args:
            observation: Serialized Observation dict.
            decision: Serialized Decision dict.
            action: Serialized Action dict.
            feedback: Serialized Feedback dict.
            reflection: Reflection dict from the cognition layer.

        Returns:
            The episode ID.
        """
        return self.episodic.store(
            observation=observation,
            decision=decision,
            action=action,
            feedback=feedback,
            reflection=reflection,
        )

    def get_recent_episodes(self, n: int = 5) -> list[dict[str, Any]]:
        """Retrieve the N most recent episodes from episodic memory.

        Args:
            n: Number of episodes to retrieve.

        Returns:
            A list of episode dicts, newest first.
        """
        return self.episodic.retrieve_recent(n)

    def get_similar_episodes(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve episodes similar to a query.

        Args:
            query: Search query string.
            top_k: Maximum number of results.

        Returns:
            A list of similar episode dicts.
        """
        return self.episodic.retrieve_by_similarity(query, top_k)

    def get_observations_text(self, n: int | None = None) -> str:
        """Format recent observations as text for LLM prompts.

        Args:
            n: Number of recent observations. None means all.

        Returns:
            A formatted string of observations.
        """
        observations = self.working.get_recent(n)
        if not observations:
            return "No observations available."
        lines = []
        for i, obs in enumerate(observations, 1):
            lines.append(f"{i}. [{obs.source}] {obs.content}")
        return "\n".join(lines)

    def get_recent_decisions_text(self, n: int = 3) -> str:
        """Format recent decision summaries from episodic memory.

        Args:
            n: Number of recent decisions to include.

        Returns:
            A formatted string of recent decisions.
        """
        episodes = self.episodic.retrieve_recent(n)
        if not episodes:
            return "No prior decisions."

        lines = []
        for i, ep in enumerate(episodes, 1):
            dec = ep.get("decision", {})
            action_name = dec.get("selected_action", "unknown")
            confidence = dec.get("confidence", 0.0)
            reasoning = dec.get("reasoning", "")[:100]
            lines.append(
                f"{i}. Action: {action_name} (confidence={confidence:.2f}) "
                f"— {reasoning}..."
            )
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Return a combined summary of both memory systems."""
        return {
            "working_memory": self.working.summary(),
            "episodic_memory": {
                "size": self.episodic.size,
                "max_entries": self.episodic.max_entries,
            },
        }
