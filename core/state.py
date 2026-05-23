"""Shared State (Blackboard) — thread-safe shared state between PCA layers.

The SharedState object acts as a blackboard pattern where each layer reads
and writes its outputs.  This decouples the layers from each other while
still allowing data to flow through the system.
"""

from __future__ import annotations

import threading
from typing import Any

from models.action import Action
from models.decision import Decision
from models.feedback import Feedback
from models.observation import Observation


class SharedState:
    """Thread-safe shared blackboard for the PCA loop.

    The SharedState holds the current observations, the latest decision,
    the latest action, and the latest feedback.  Each layer reads what it
    needs and writes its output.  A lock ensures thread safety if the
    loop is ever extended to run layers concurrently.

    Attributes:
        observations: Current list of filtered observations.
        decision: The most recent decision from the cognition layer.
        action: The most recent action from the action layer.
        feedback: The most recent feedback from the environment.
        iteration: The current loop iteration number.
        metrics: Accumulated metrics for the session.
    """

    def __init__(self) -> None:
        """Initialize an empty shared state."""
        self._lock = threading.Lock()
        self.observations: list[Observation] = []
        self.decision: Decision | None = None
        self.action: Action | None = None
        self.feedback: Feedback | None = None
        self.iteration: int = 0
        self.metrics: dict[str, Any] = {
            "decisions_made": 0,
            "actions_executed": 0,
            "total_confidence": 0.0,
            "action_distribution": {},
            "start_time": None,
            "errors": 0,
        }
        self._running: bool = False

    def set_observations(self, observations: list[Observation]) -> None:
        """Update the current observations.

        Args:
            observations: The new list of observations.
        """
        with self._lock:
            self.observations = observations

    def get_observations(self) -> list[Observation]:
        """Return a copy of the current observations."""
        with self._lock:
            return list(self.observations)

    def set_decision(self, decision: Decision) -> None:
        """Record a new decision from the cognition layer.

        Args:
            decision: The Decision to store.
        """
        with self._lock:
            self.decision = decision
            self.metrics["decisions_made"] += 1
            self.metrics["total_confidence"] += decision.confidence

    def get_decision(self) -> Decision | None:
        """Return the latest decision."""
        with self._lock:
            return self.decision

    def set_action(self, action: Action) -> None:
        """Record a new action from the action layer.

        Args:
            action: The Action to store.
        """
        with self._lock:
            self.action = action
            self.metrics["actions_executed"] += 1
            action_type = action.action_type
            dist = self.metrics["action_distribution"]
            dist[action_type] = dist.get(action_type, 0) + 1

    def get_action(self) -> Action | None:
        """Return the latest action."""
        with self._lock:
            return self.action

    def set_feedback(self, feedback: Feedback) -> None:
        """Record feedback from the environment.

        Args:
            feedback: The Feedback to store.
        """
        with self._lock:
            self.feedback = feedback

    def get_feedback(self) -> Feedback | None:
        """Return the latest feedback."""
        with self._lock:
            return self.feedback

    def increment_iteration(self) -> int:
        """Increment the iteration counter and return the new value."""
        with self._lock:
            self.iteration += 1
            return self.iteration

    def get_iteration(self) -> int:
        """Return the current iteration number."""
        with self._lock:
            return self.iteration

    def record_error(self) -> None:
        """Increment the error counter."""
        with self._lock:
            self.metrics["errors"] += 1

    @property
    def running(self) -> bool:
        """Whether the PCA loop is currently running."""
        return self._running

    @running.setter
    def running(self, value: bool) -> None:
        self._running = value

    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of the current metrics."""
        with self._lock:
            metrics = dict(self.metrics)
            if metrics["decisions_made"] > 0:
                metrics["avg_confidence"] = (
                    metrics["total_confidence"] / metrics["decisions_made"]
                )
            else:
                metrics["avg_confidence"] = 0.0
            metrics["iteration"] = self.iteration
            return metrics

    def summary(self) -> dict[str, Any]:
        """Return a full summary of the shared state."""
        with self._lock:
            return {
                "iteration": self.iteration,
                "observations_count": len(self.observations),
                "has_decision": self.decision is not None,
                "has_action": self.action is not None,
                "has_feedback": self.feedback is not None,
                "metrics": self.get_metrics(),
            }
