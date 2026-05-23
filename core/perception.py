"""Perception Layer — observe, filter, and normalize environmental inputs.

The perception layer is the agent's sensory interface.  It pulls raw
observations from the environment, applies attention filtering (relevance
scoring + recency weighting), normalizes them into structured Observation
objects, and maintains a rolling observation window.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any

from core.state import SharedState
from environment.base import BaseEnvironment
from memory.memory_manager import MemoryManager
from models.observation import Observation
from utils.logger import AgentLogger


class PerceptionLayer:
    """The Perception layer of the PCA loop.

    Responsible for:
    - Observing the environment and collecting raw events.
    - Filtering observations by relevance and recency.
    - Normalizing raw events into structured Observation objects.
    - Maintaining a rolling observation window.
    - Injecting feedback-driven observations back into the pipeline.

    Attributes:
        observation_window: Rolling buffer of recent filtered observations.
        env: The environment being observed.
        memory: The memory manager for storing observations.
        state: The shared blackboard.
        logger: Agent logger instance.
    """

    def __init__(
        self,
        environment: BaseEnvironment,
        memory: MemoryManager,
        state: SharedState,
        config: dict[str, Any],
        logger: AgentLogger | None = None,
    ) -> None:
        """Initialize the perception layer.

        Args:
            environment: The environment to observe.
            memory: The memory manager for storing observations.
            state: The shared state blackboard.
            config: Full agent configuration dictionary.
            logger: Optional logger instance.
        """
        self.env = environment
        self.memory = memory
        self.state = state
        self.config = config

        agent_cfg = config.get("agent", {})
        window_size = agent_cfg.get("observation_window_size", 10)
        self.observation_window: deque[Observation] = deque(maxlen=window_size)

        self.logger = logger or AgentLogger(
            name="Perception",
            log_file=config.get("logging", {}).get("file"),
            level=config.get("logging", {}).get("level", "INFO"),
        )

        self._relevance_threshold: float = 0.15

    def perceive(self) -> list[Observation]:
        """Execute one perception cycle: observe → filter → normalize → store.

        This is the main entry point called by the PCA loop on each
        iteration.  It pulls new observations from the environment,
        applies attention filtering, normalizes them, stores them in
        the observation window and working memory, and updates the
        shared state.

        Returns:
            The filtered list of observations for this cycle.
        """
        # Step 1: Observe — pull raw events from the environment
        raw_observations = self.observe()

        # Step 2: Filter — apply attention filtering
        filtered = self.filter_observations(raw_observations)

        # Step 3: Normalize — ensure consistent structure
        normalized = self.normalize(filtered)

        # Step 4: Update observation window
        for obs in normalized:
            self.observation_window.append(obs)

        # Step 5: Store in working memory
        for obs in normalized:
            self.memory.add_observation(obs)

        # Step 6: Update shared state
        self.state.set_observations(list(self.observation_window))

        self.logger.info(
            f"Perceived {len(normalized)} observations "
            f"(window size: {len(self.observation_window)})",
            raw_count=len(raw_observations),
            filtered_count=len(normalized),
        )

        return normalized

    def observe(self) -> list[Observation]:
        """Pull current observations from the environment.

        Also incorporates any feedback-driven observations from the
        previous action cycle.

        Returns:
            A list of raw Observation objects from the environment.
        """
        observations = self.env.observe()

        # Inject feedback observations if available
        feedback = self.state.get_feedback()
        if feedback and feedback.new_observations:
            for obs_text in feedback.new_observations:
                observations.append(
                    Observation(
                        timestamp=datetime.now(),
                        source="feedback",
                        content=obs_text,
                        relevance_score=0.7,
                        metadata={"action_id": feedback.action_id},
                    )
                )

        return observations

    def filter_observations(
        self, observations: list[Observation]
    ) -> list[Observation]:
        """Apply attention filtering to prioritize relevant observations.

        The filtering strategy combines two factors:
        1. **Relevance score** — each observation's inherent relevance.
        2. **Recency weighting** — more recent observations get a boost.

        Observations below the relevance threshold are dropped unless they
        come from high-priority sources (e.g. deadline_warning, task_arrival).

        Args:
            observations: The raw observations to filter.

        Returns:
            The filtered list of observations.
        """
        if not observations:
            return []

        now = datetime.now()
        filtered = []

        for obs in observations:
            # High-priority sources always pass
            high_priority_sources = {
                "deadline_warning",
                "task_arrival",
                "task_completed",
                "action_feedback",
                "feedback",
            }

            if obs.source in high_priority_sources:
                obs.relevance_score = max(obs.relevance_score, 0.7)
                filtered.append(obs)
                continue

            # Apply recency weighting
            age_seconds = (now - obs.timestamp).total_seconds()
            recency_factor = max(0.0, 1.0 - age_seconds / 300.0)  # 5-min half-life

            # Combined score
            combined_score = 0.7 * obs.relevance_score + 0.3 * recency_factor

            if combined_score >= self._relevance_threshold:
                obs.relevance_score = combined_score
                filtered.append(obs)

        # Sort by relevance (highest first)
        filtered.sort(key=lambda o: o.relevance_score, reverse=True)

        return filtered

    def normalize(self, observations: list[Observation]) -> list[Observation]:
        """Normalize observations into a consistent format.

        Ensures all observations have proper timestamps, sources, and
        metadata.  Truncates excessively long content and standardizes
        source names.

        Args:
            observations: The observations to normalize.

        Returns:
            The normalized observations.
        """
        normalized = []

        for obs in observations:
            # Ensure timestamp is set
            if obs.timestamp is None:
                obs.timestamp = datetime.now()

            # Truncate very long content
            if len(obs.content) > 500:
                obs.content = obs.content[:497] + "..."

            # Normalize source name
            obs.source = obs.source.lower().strip().replace(" ", "_")

            # Ensure metadata is a dict
            if obs.metadata is None:
                obs.metadata = {}

            # Clamp relevance score
            obs.relevance_score = max(0.0, min(1.0, obs.relevance_score))

            normalized.append(obs)

        return normalized

    def get_window_summary(self) -> str:
        """Return a text summary of the current observation window.

        Useful for providing context to the cognition layer's LLM prompts.

        Returns:
            A formatted string of the current observations.
        """
        if not self.observation_window:
            return "No observations in window."

        lines = []
        for i, obs in enumerate(reversed(self.observation_window), 1):
            lines.append(f"{i}. {obs}")

        return "\n".join(lines)
