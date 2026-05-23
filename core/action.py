"""Action Layer — execute actions, collect feedback, and log outcomes.

The action layer receives the selected action from the cognition layer,
dispatches it to the appropriate handler, collects feedback from the
environment, and logs the complete action trace.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.state import SharedState
from environment.base import BaseEnvironment
from models.action import Action, ActionType
from models.decision import Decision
from models.feedback import Feedback
from utils.logger import AgentLogger


class ActionLayer:
    """The Action layer of the PCA loop.

    Responsible for:
    - Receiving the selected action from the cognition layer's Decision.
    - Constructing an Action object with the correct type and parameters.
    - Dispatching the action to the environment.
    - Collecting feedback from the environment.
    - Logging the action and feedback for traceability.
    - Updating the shared state with the action and feedback.

    Attributes:
        env: The environment to execute actions in.
        state: The shared state blackboard.
        logger: Agent logger instance.
        dry_run: If True, actions are logged but not actually executed.
    """

    def __init__(
        self,
        environment: BaseEnvironment,
        state: SharedState,
        config: dict[str, Any],
        logger: AgentLogger | None = None,
        dry_run: bool = False,
    ) -> None:
        """Initialize the action layer.

        Args:
            environment: The environment to execute actions in.
            state: The shared state blackboard.
            config: Full agent configuration dictionary.
            logger: Optional logger instance.
            dry_run: If True, simulate actions without executing them.
        """
        self.env = environment
        self.state = state
        self.config = config
        self.dry_run = dry_run

        self.logger = logger or AgentLogger(
            name="Action",
            log_file=config.get("logging", {}).get("file"),
            level=config.get("logging", {}).get("level", "INFO"),
        )

        self._action_history: list[dict[str, Any]] = []

    def act(self, decision: Decision) -> Feedback:
        """Execute the selected action from a Decision.

        Constructs an Action object from the decision, dispatches it to
        the environment, collects feedback, and updates the shared state.

        Args:
            decision: The Decision containing the selected action.

        Returns:
            The Feedback from the environment.
        """
        # Construct the Action object
        action = self._construct_action(decision)

        # Update shared state with the action
        self.state.set_action(action)

        if self.dry_run:
            # In dry-run mode, simulate positive feedback without executing
            feedback = Feedback(
                action_id=action.action_id,
                success=True,
                signal="neutral",
                message=f"[DRY RUN] Action {action.action_type} would have been executed.",
                new_observations=[],
            )
            self.logger.info(
                f"[DRY RUN] {action.action_type}: {action.params}",
                action_id=action.action_id,
            )
        else:
            # Execute the action in the environment
            feedback = self._execute_action(action)

        # Update shared state with feedback
        self.state.set_feedback(feedback)

        # Log the action + feedback
        self._log_action(action, feedback)

        # Add to history
        self._action_history.append({
            "action": action.to_dict(),
            "feedback": feedback.to_dict(),
            "decision_id": decision.decision_id,
            "timestamp": datetime.now().isoformat(),
        })

        return feedback

    def _construct_action(self, decision: Decision) -> Action:
        """Construct an Action object from a Decision.

        Parses the decision's selected_action and extracts parameters from
        the candidate actions to build a well-formed Action.

        Args:
            decision: The Decision to convert.

        Returns:
            A constructed Action object.
        """
        action_type = decision.selected_action
        params: dict[str, Any] = {}

        # Try to find the matching candidate action for parameters
        for candidate in decision.candidate_actions:
            if candidate.get("action_type") == action_type or \
               candidate.get("name") == action_type:
                params = candidate.get("params", {})
                break

        # Ensure the action type is valid
        valid_types = {t.value for t in ActionType}
        if action_type not in valid_types:
            # Try to map common action names to types
            action_type = self._infer_action_type(action_type, decision)

        return Action(
            action_type=action_type,
            params=params,
            timestamp=datetime.now(),
        )

    def _infer_action_type(self, action_name: str, decision: Decision) -> str:
        """Infer the action type from an action name or context.

        Args:
            action_name: The raw action name from the decision.
            decision: The full decision for context.

        Returns:
            A valid ActionType value.
        """
        name_lower = action_name.lower()

        if "start" in name_lower or "begin" in name_lower:
            return ActionType.TOOL_USE.value
            params = decision.candidate_actions[0].get("params", {}) if decision.candidate_actions else {}
            # We'll add the tool param in _construct_action
        elif "defer" in name_lower or "postpone" in name_lower:
            return ActionType.TOOL_USE.value
        elif "request" in name_lower or "query" in name_lower or "info" in name_lower:
            if "query_environment" in name_lower:
                return ActionType.QUERY_ENVIRONMENT.value
            return ActionType.TOOL_USE.value
        elif "escalate" in name_lower:
            return ActionType.ESCALATE.value
        elif "wait" in name_lower or "hold" in name_lower:
            return ActionType.WAIT.value
        elif "respond" in name_lower or "message" in name_lower or "reply" in name_lower:
            return ActionType.RESPOND.value
        else:
            # Default to respond for unknown action names
            return ActionType.RESPOND.value

    def _execute_action(self, action: Action) -> Feedback:
        """Execute an action in the environment and return feedback.

        Handles errors gracefully — if the environment throws an exception,
        it is caught and converted into negative feedback.

        Args:
            action: The Action to execute.

        Returns:
            A Feedback object describing the outcome.
        """
        try:
            feedback = self.env.process_action(action)
            self.logger.info(
                f"Executed {action.action_type}: "
                f"{'OK' if feedback.success else 'FAIL'}",
                action_id=action.action_id,
                action_type=action.action_type,
                success=feedback.success,
            )
            return feedback
        except Exception as exc:
            self.logger.error(
                f"Action execution error: {exc}",
                action_id=action.action_id,
                action_type=action.action_type,
            )
            self.state.record_error()
            return Feedback(
                action_id=action.action_id,
                success=False,
                signal="negative",
                message=f"Action execution failed: {exc}",
                new_observations=[],
            )

    def _log_action(self, action: Action, feedback: Feedback) -> None:
        """Log the action and its feedback.

        Args:
            action: The Action that was executed.
            feedback: The Feedback received.
        """
        log_cfg = self.config.get("logging", {})
        if log_cfg.get("log_decisions", True):
            self.logger.log_action(
                action=action.to_dict(),
                feedback=feedback.to_dict(),
            )

    @property
    def action_count(self) -> int:
        """Total number of actions executed."""
        return len(self._action_history)

    def get_history(self, n: int | None = None) -> list[dict[str, Any]]:
        """Return recent action history.

        Args:
            n: Number of recent actions to return. None for all.

        Returns:
            A list of action history entries.
        """
        if n is None:
            return list(self._action_history)
        return list(self._action_history[-n:])
