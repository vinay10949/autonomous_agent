"""Abstract base class for environments.

Defines the interface that all environments (simulated, real, custom) must
implement.  The environment is the agent's window to the outside world — it
provides observations and processes the agent's actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from models.action import Action
from models.feedback import Feedback
from models.observation import Observation


class BaseEnvironment(ABC):
    """Abstract interface for agent environments.

    Every environment must implement three core operations:
    1. ``observe`` — return the current set of observations.
    2. ``process_action`` — execute an action and return feedback.
    3. ``reset`` — return the environment to its initial state.

    The environment is polled by the perception layer on each PCA cycle.
    Actions submitted by the action layer are processed here, and the
    resulting feedback is returned for the next perception cycle.
    """

    @abstractmethod
    def observe(self) -> list[Observation]:
        """Return the current observations from the environment.

        This method is called once per PCA cycle by the perception layer.
        It should return all new events, state changes, or signals that
        have occurred since the last observation.

        Returns:
            A list of Observation objects representing current environmental state.
        """

    @abstractmethod
    def process_action(self, action: Action) -> Feedback:
        """Execute an action in the environment and return feedback.

        Args:
            action: The Action to execute.

        Returns:
            A Feedback object describing the outcome of the action.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment to its initial state."""

    @abstractmethod
    def get_state_summary(self) -> dict[str, Any]:
        """Return a summary of the current environment state.

        Used by the cognition layer to understand the environment context
        without needing to parse individual observations.

        Returns:
            A dictionary summarizing the environment state.
        """

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether the environment is still producing events.

        Returns:
            True if the environment has more events to produce, False otherwise.
        """
