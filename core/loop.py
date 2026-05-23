"""PCA Loop Controller — orchestrates the continuous Perception → Cognition → Action cycle.

The PCALoop is the heart of the autonomous agent.  It runs the three layers
in sequence on each iteration, handles timing, graceful shutdown, and
metrics collection.
"""

from __future__ import annotations

import signal
import time
from typing import Any

from core.action import ActionLayer
from core.cognition import CognitionLayer
from core.perception import PerceptionLayer
from core.state import SharedState
from models.decision import Decision
from utils.logger import AgentLogger


class PCALoop:
    """The main PCA loop controller.

    Runs the continuous cycle:
        perceive() → think() → act() → process_feedback()

    The loop respects the configured interval (not busy-waiting), has a
    maximum iteration guard, and supports graceful shutdown via SIGINT/SIGTERM.

    Attributes:
        perception: The Perception layer.
        cognition: The Cognition layer.
        action: The Action layer.
        state: The shared state blackboard.
        interval: Seconds between loop iterations.
        max_iterations: Maximum number of iterations before stopping.
        logger: Agent logger instance.
    """

    def __init__(
        self,
        perception: PerceptionLayer,
        cognition: CognitionLayer,
        action: ActionLayer,
        state: SharedState,
        config: dict[str, Any],
        logger: AgentLogger | None = None,
    ) -> None:
        """Initialize the PCA loop.

        Args:
            perception: The Perception layer instance.
            cognition: The Cognition layer instance.
            action: The Action layer instance.
            state: The shared state blackboard.
            config: Full agent configuration dictionary.
            logger: Optional logger instance.
        """
        self.perception = perception
        self.cognition = cognition
        self.action = action
        self.state = state

        agent_cfg = config.get("agent", {})
        self.interval = agent_cfg.get("loop_interval_seconds", 2.0)
        self.max_iterations = agent_cfg.get("max_iterations", 50)
        self.agent_name = agent_cfg.get("name", "AutonomousAgent")

        self.logger = logger or AgentLogger(
            name="PCALoop",
            log_file=config.get("logging", {}).get("file"),
            level=config.get("logging", {}).get("level", "INFO"),
        )

        self._shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(
            f"Received signal {signum}. Requesting graceful shutdown..."
        )
        self._shutdown_requested = True

    def request_shutdown(self) -> None:
        """Request a graceful shutdown of the loop."""
        self._shutdown_requested = True

    def run(self) -> dict[str, Any]:
        """Run the PCA loop until completion or shutdown.

        The loop executes the following on each iteration:
        1. Check if shutdown has been requested.
        2. Check if max iterations have been reached.
        3. Execute the perceive → think → act → reflect cycle.
        4. Wait for the configured interval.

        Returns:
            A metrics dictionary summarizing the session.
        """
        self.state.running = True
        self.state.metrics["start_time"] = time.time()

        self.logger.info(
            f"Starting PCA loop for '{self.agent_name}' "
            f"(max_iterations={self.max_iterations}, "
            f"interval={self.interval}s)"
        )

        try:
            while not self._shutdown_requested:
                iteration = self.state.increment_iteration()

                # Check max iterations
                if iteration > self.max_iterations:
                    self.logger.info(
                        f"Max iterations ({self.max_iterations}) reached. Stopping."
                    )
                    break

                # Execute one PCA cycle
                self._run_cycle(iteration)

                # Wait for the next cycle
                time.sleep(self.interval)

        except Exception as exc:
            self.logger.error(f"PCA loop crashed: {exc}")
            self.state.record_error()
        finally:
            self.state.running = False
            self._print_final_metrics()

        return self.state.get_metrics()

    def _run_cycle(self, iteration: int) -> None:
        """Execute one complete PCA cycle.

        Args:
            iteration: The current iteration number.
        """
        cycle_start = time.time()

        # 1. PERCEPTION
        observations = self.perception.perceive()

        # 2. COGNITION
        decision = self.cognition.think()

        # 3. ACTION
        feedback = self.action.act(decision)

        # 4. REFLECTION
        action_obj = self.state.get_action()
        if action_obj is not None:
            self.cognition.reflect(decision, action_obj, feedback)

        cycle_duration = time.time() - cycle_start

        self.logger.info(
            f"Cycle {iteration} completed in {cycle_duration:.2f}s "
            f"— Action: {decision.selected_action} "
            f"(confidence={decision.confidence:.2f}, "
            f"feedback={'OK' if feedback.success else 'FAIL'})",
            iteration=iteration,
            cycle_duration=cycle_duration,
            action=decision.selected_action,
            confidence=decision.confidence,
            feedback_success=feedback.success,
        )

    def _print_final_metrics(self) -> None:
        """Print final session metrics."""
        metrics = self.state.get_metrics()
        start_time = metrics.get("start_time", time.time())
        elapsed = time.time() - start_time if start_time else 0.0

        decisions = metrics.get("decisions_made", 0)
        actions = metrics.get("actions_executed", 0)
        avg_conf = metrics.get("avg_confidence", 0.0)
        action_dist = metrics.get("action_distribution", {})
        errors = metrics.get("errors", 0)

        self.logger.info("=" * 50)
        self.logger.info("SESSION SUMMARY")
        self.logger.info(f"  Total iterations:  {metrics.get('iteration', 0)}")
        self.logger.info(f"  Decisions made:    {decisions}")
        self.logger.info(f"  Actions executed:  {actions}")
        self.logger.info(f"  Avg confidence:    {avg_conf:.2f}")
        self.logger.info(f"  Errors:            {errors}")
        self.logger.info(f"  Elapsed time:      {elapsed:.1f}s")
        if decisions > 0 and elapsed > 0:
            self.logger.info(f"  Decisions/min:     {decisions / (elapsed / 60):.1f}")
        self.logger.info(f"  Action distribution: {action_dist}")
        self.logger.info("=" * 50)
