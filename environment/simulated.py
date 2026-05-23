"""Simulated Environment — generates events and processes actions for testing.

The simulated environment provides a controllable, reproducible setting for
the agent to operate in.  It generates events according to the configured
scenario and responds to agent actions with realistic outcomes.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from environment.base import BaseEnvironment
from environment.scenarios.task_orchestration import TaskOrchestrationScenario
from models.action import Action, ActionType
from models.feedback import Feedback, FeedbackSignal
from models.observation import Observation


class SimulatedEnvironment(BaseEnvironment):
    """A simulated environment driven by a pluggable scenario.

    The environment maintains an internal event queue that is populated by
    the scenario.  On each ``observe()`` call, it returns pending events
    as Observation objects.  When the agent performs an action via
    ``process_action()``, the scenario determines the outcome and generates
    appropriate feedback.

    Attributes:
        scenario: The active scenario generating events and handling actions.
        _tick: Internal counter incremented each observation cycle.
        _event_queue: Pending observations not yet consumed.
        _active: Whether the environment is still producing events.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the simulated environment.

        Args:
            config: The full agent configuration.  The 'environment' section
                determines which scenario to load.
        """
        env_config = config.get("environment", {})
        scenario_name = env_config.get("scenario", "task_orchestration")

        if scenario_name == "task_orchestration":
            self.scenario = TaskOrchestrationScenario(config)
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        self._tick: int = 0
        self._event_queue: list[Observation] = []
        self._active: bool = True
        self._started_tasks: set[str] = set()

    def observe(self) -> list[Observation]:
        """Generate new observations from the scenario.

        On each call, the scenario is asked to produce new events based
        on the current tick.  These events are converted to Observation
        objects and returned.  The internal tick counter is advanced.

        Returns:
            A list of new Observation objects.
        """
        self._tick += 1

        # Ask scenario to generate events for this tick
        new_events = self.scenario.generate_events(self._tick)

        # Convert raw events to Observation objects
        observations = []
        for event in new_events:
            obs = Observation(
                timestamp=datetime.now(),
                source=event.get("source", "environment"),
                content=event.get("content", ""),
                relevance_score=event.get("relevance", 0.5),
                metadata=event.get("metadata", {}),
            )
            observations.append(obs)
            self._event_queue.append(obs)

        # Check if the scenario is done
        if self.scenario.is_complete():
            self._active = False

        return observations

    def process_action(self, action: Action) -> Feedback:
        """Process an agent action and return feedback.

        The action is delegated to the scenario for evaluation.  The
        scenario determines the outcome and may generate follow-up events.

        Args:
            action: The Action to execute.

        Returns:
            A Feedback object describing the outcome.
        """
        result = self.scenario.handle_action(action)

        # Any follow-up observations from the action
        new_obs = result.get("new_observations", [])
        for obs_text in new_obs:
            self._event_queue.append(
                Observation(
                    timestamp=datetime.now(),
                    source="action_feedback",
                    content=obs_text,
                    relevance_score=0.7,
                    metadata={"action_id": action.action_id},
                )
            )

        feedback = Feedback(
            action_id=action.action_id,
            success=result.get("success", False),
            signal=result.get("signal", "neutral"),
            message=result.get("message", "Action processed."),
            new_observations=new_obs,
        )
        return feedback

    def reset(self) -> None:
        """Reset the environment to its initial state."""
        self._tick = 0
        self._event_queue = []
        self._active = True
        self._started_tasks = set()
        self.scenario.reset()

    def get_state_summary(self) -> dict[str, Any]:
        """Return a summary of the current environment state."""
        return {
            "tick": self._tick,
            "active": self._active,
            "pending_observations": len(self._event_queue),
            "scenario_summary": self.scenario.get_summary(),
        }

    @property
    def is_active(self) -> bool:
        """Whether the environment is still producing events."""
        return self._active
