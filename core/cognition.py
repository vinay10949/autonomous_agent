"""Cognition Layer — the agent's reasoning and decision engine.

The cognition layer is the "brain" of the autonomous agent.  It receives
structured observations from the perception layer, uses the LLM to assess
the situation, generate candidate actions, evaluate them, and select the
best action.  It also implements self-reflection and uncertainty handling.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.state import SharedState
from llm.client import LLMClient
from llm.prompts import PromptTemplates
from memory.memory_manager import MemoryManager
from models.action import Action, ActionType
from models.decision import Decision
from utils.logger import AgentLogger


class CognitionLayer:
    """The Cognition layer of the PCA loop — the agent's decision engine.

    Implements the full decision pipeline:
    1. Situation Assessment — summarize the current state.
    2. Action Generation — propose candidate actions via LLM.
    3. Action Evaluation — score each action on multiple dimensions.
    4. Confidence Check — determine if confidence is sufficient to act.
    5. Action Selection — pick the highest-scoring action.
    6. Reflection — post-decision self-reflection for auditability.

    Attributes:
        llm: The LLM client for reasoning calls.
        memory: The memory manager for context retrieval.
        state: The shared state blackboard.
        confidence_threshold: Below this, the agent gathers more info.
        logger: Agent logger instance.
    """

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryManager,
        state: SharedState,
        config: dict[str, Any],
        logger: AgentLogger | None = None,
    ) -> None:
        """Initialize the cognition layer.

        Args:
            llm: The LM Studio LLM client.
            memory: The memory manager.
            state: The shared state blackboard.
            config: Full agent configuration dictionary.
            logger: Optional logger instance.
        """
        self.llm = llm
        self.memory = memory
        self.state = state
        self.config = config

        agent_cfg = config.get("agent", {})
        self.confidence_threshold = agent_cfg.get("confidence_threshold", 0.6)
        self.agent_name = agent_cfg.get("name", "AutonomousAgent")

        self.logger = logger or AgentLogger(
            name="Cognition",
            log_file=config.get("logging", {}).get("file"),
            level=config.get("logging", {}).get("level", "INFO"),
        )

        self._last_situation_summary: str = ""

    def think(self) -> Decision:
        """Execute one cognition cycle: assess → generate → evaluate → select.

        This is the main entry point called by the PCA loop.  It orchestrates
        the full decision pipeline and returns a Decision object containing
        the selected action and full reasoning trace.

        Returns:
            A Decision object with the selected action, confidence, and reasoning.
        """
        # Step 1: Assess the current situation
        situation = self.assess_situation()

        # Step 2: Generate candidate actions
        candidates = self.generate_candidate_actions(situation)

        if not candidates:
            # No candidates — default to waiting
            return Decision(
                timestamp=datetime.now(),
                situation_summary=situation.get("situation_summary", "No situation data."),
                candidate_actions=[],
                selected_action="wait",
                reasoning="No candidate actions were generated. Defaulting to wait.",
                confidence=0.3,
            )

        # Step 3: Evaluate and score each candidate
        evaluation = self.evaluate_actions(situation, candidates)

        # Step 4: Select the best action (with confidence check)
        decision = self.select_action(situation, candidates, evaluation)

        # Step 5: Store the decision in shared state
        self.state.set_decision(decision)

        self.logger.info(
            f"Decision: {decision.selected_action} "
            f"(confidence={decision.confidence:.2f})",
            decision_id=decision.decision_id,
            reasoning_summary=decision.reasoning[:200],
        )

        return decision

    def assess_situation(self) -> dict[str, Any]:
        """Assess the current situation using observations and memory.

        Uses the LLM to synthesize the current observations, working memory
        context, and recent decisions into a structured situation assessment.

        Returns:
            A dictionary with situation_summary, urgent_items, patterns, risks.
        """
        observations_text = self.memory.get_observations_text(n=10)
        working_memory_text = self._format_working_memory()
        recent_decisions_text = self.memory.get_recent_decisions_text(n=3)

        prompt = PromptTemplates.situation_assessment(
            observations=observations_text,
            working_memory=working_memory_text,
            recent_decisions=recent_decisions_text,
        )

        try:
            result = self.llm.generate_json(
                prompt=prompt,
                system_msg=(
                    f"You are {self.agent_name}, an autonomous decision-making agent. "
                    "Assess the current situation based on the observations provided."
                ),
                temperature=0.4,
            )
        except (ValueError, RuntimeError) as exc:
            self.logger.warning(f"Situation assessment failed: {exc}")
            result = {
                "situation_summary": f"Assessment incomplete due to LLM error: {exc}",
                "urgent_items": [],
                "patterns": [],
                "risks": [],
                "overall_urgency": "medium",
            }

        self._last_situation_summary = result.get("situation_summary", "")
        return result

    def generate_candidate_actions(self, situation: dict[str, Any]) -> list[dict]:
        """Generate candidate actions based on the situation assessment.

        Uses the LLM to propose 3-5 candidate actions that the agent could
        take, along with rationales for each.

        Args:
            situation: The situation assessment from assess_situation().

        Returns:
            A list of candidate action dictionaries.
        """
        from environment.simulated import SimulatedEnvironment

        # Get available tasks from the environment if it's simulated
        available_tasks = "No task information available."
        env = getattr(self, "_env_ref", None)
        if env and hasattr(env, "scenario") and hasattr(env.scenario, "get_available_tasks_text"):
            available_tasks = env.scenario.get_available_tasks_text()

        prompt = PromptTemplates.action_generation(
            situation_summary=situation.get("situation_summary", "Unknown"),
            urgent_items="\n".join(situation.get("urgent_items", [])),
            available_tasks=available_tasks,
        )

        try:
            result = self.llm.generate_json(
                prompt=prompt,
                system_msg=(
                    f"You are {self.agent_name}. Generate candidate actions "
                    "based on the current situation."
                ),
                temperature=0.6,
            )
            return result.get("candidates", [])
        except (ValueError, RuntimeError) as exc:
            self.logger.warning(f"Action generation failed: {exc}")
            # Fallback: generate a simple wait action
            return [
                {
                    "name": "wait-for-info",
                    "description": "Wait for more information from the environment.",
                    "action_type": "wait",
                    "params": {},
                    "rationale": "LLM action generation failed; waiting is the safest default.",
                }
            ]

    def evaluate_actions(
        self,
        situation: dict[str, Any],
        candidates: list[dict],
    ) -> dict[str, Any]:
        """Score each candidate action on multiple dimensions.

        Uses the LLM to evaluate each action on effectiveness, risk,
        information gain, and alignment, then compute a total score.

        Args:
            situation: The situation assessment.
            candidates: The list of candidate action dicts.

        Returns:
            A dictionary with evaluations list, confidence, and reasoning.
        """
        candidates_text = "\n".join(
            f"- {c.get('name', 'unknown')}: {c.get('description', 'no description')} "
            f"(type={c.get('action_type', 'unknown')}, "
            f"params={c.get('params', {})})"
            for c in candidates
        )

        prompt = PromptTemplates.action_evaluation(
            situation_summary=situation.get("situation_summary", "Unknown"),
            candidate_actions=candidates_text,
        )

        try:
            result = self.llm.generate_json(
                prompt=prompt,
                system_msg=(
                    f"You are {self.agent_name}. Evaluate candidate actions "
                    "by scoring them on effectiveness, risk, information gain, and alignment."
                ),
                temperature=0.3,
            )
            return result
        except (ValueError, RuntimeError) as exc:
            self.logger.warning(f"Action evaluation failed: {exc}")
            # Fallback: assign equal scores
            evaluations = []
            for candidate in candidates:
                evaluations.append({
                    "name": candidate.get("name", "unknown"),
                    "effectiveness": 0.5,
                    "risk": 0.5,
                    "information_gain": 0.5,
                    "alignment": 0.5,
                    "total_score": 0.5,
                })
            return {
                "evaluations": evaluations,
                "confidence": 0.3,
                "reasoning": f"Evaluation LLM call failed: {exc}",
            }

    def select_action(
        self,
        situation: dict[str, Any],
        candidates: list[dict],
        evaluation: dict[str, Any],
    ) -> Decision:
        """Select the best action based on evaluation scores and confidence.

        Implements the confidence threshold check: if the overall confidence
        is below the threshold, the agent defaults to gathering more
        information (query_environment) rather than taking a risky action.

        Args:
            situation: The situation assessment.
            candidates: The candidate actions.
            evaluation: The evaluation results with scores.

        Returns:
            A Decision object with the selected action and full reasoning.
        """
        evaluations = evaluation.get("evaluations", [])
        confidence = evaluation.get("confidence", 0.0)
        eval_reasoning = evaluation.get("reasoning", "")

        # Merge scores into candidates
        score_map = {e["name"]: e for e in evaluations}
        for candidate in candidates:
            name = candidate.get("name", "")
            if name in score_map:
                candidate["scores"] = score_map[name]

        # Sort candidates by total_score (descending)
        scored_candidates = sorted(
            [c for c in candidates if "scores" in c],
            key=lambda c: c.get("scores", {}).get("total_score", 0.0),
            reverse=True,
        )

        # Confidence check
        if confidence < self.confidence_threshold and scored_candidates:
            # Low confidence — check if we should gather more info
            uncertainty_result = self._check_uncertainty(
                situation, scored_candidates[0], confidence
            )
            if uncertainty_result.get("recommendation") == "gather_info":
                return Decision(
                    timestamp=datetime.now(),
                    situation_summary=situation.get("situation_summary", ""),
                    candidate_actions=[
                        {**c, "scores": c.get("scores", {})} for c in candidates
                    ],
                    selected_action="query_environment",
                    reasoning=(
                        f"Confidence ({confidence:.2f}) below threshold "
                        f"({self.confidence_threshold}). "
                        f"Recommendation: {uncertainty_result.get('reasoning', 'gather info')}. "
                        f"Original top action was: {scored_candidates[0].get('name', 'unknown')}"
                    ),
                    confidence=confidence,
                )
            elif uncertainty_result.get("recommendation") == "wait":
                return Decision(
                    timestamp=datetime.now(),
                    situation_summary=situation.get("situation_summary", ""),
                    candidate_actions=[
                        {**c, "scores": c.get("scores", {})} for c in candidates
                    ],
                    selected_action="wait",
                    reasoning=(
                        f"Confidence ({confidence:.2f}) below threshold. "
                        f"Waiting for situation to evolve. "
                        f"{uncertainty_result.get('reasoning', '')}"
                    ),
                    confidence=confidence,
                )

        # Select top-scored action
        if scored_candidates:
            best = scored_candidates[0]
            action_type = best.get("action_type", "respond")
            params = best.get("params", {})

            # Map candidate to action type
            selected_name = best.get("name", "unknown")

            # If the action_type is tool_use, add the tool name to params
            if action_type == "tool_use" and "tool" not in params:
                # Infer tool from the candidate name
                if "start" in selected_name:
                    params["tool"] = "start_task"
                elif "defer" in selected_name:
                    params["tool"] = "defer_task"
                elif "info" in selected_name or "query" in selected_name:
                    params["tool"] = "request_info"
                else:
                    params["tool"] = selected_name

            return Decision(
                timestamp=datetime.now(),
                situation_summary=situation.get("situation_summary", ""),
                candidate_actions=[
                    {**c, "scores": c.get("scores", {})} for c in candidates
                ],
                selected_action=action_type,
                reasoning=(
                    f"{eval_reasoning}\n\n"
                    f"Selected '{selected_name}' (score="
                    f"{best.get('scores', {}).get('total_score', 0.0):.2f}) "
                    f"with confidence {confidence:.2f}."
                ),
                confidence=confidence,
            )

        # No scored candidates — default to wait
        return Decision(
            timestamp=datetime.now(),
            situation_summary=situation.get("situation_summary", ""),
            candidate_actions=candidates,
            selected_action="wait",
            reasoning="No scored candidates available. Defaulting to wait.",
            confidence=0.0,
        )

    def reflect(
        self,
        decision: Decision,
        action: Action,
        feedback: Any,
    ) -> dict[str, Any]:
        """Perform post-decision reflection for auditability.

        After an action is executed and feedback is received, this method
        uses the LLM to reflect on whether the decision was sound, what
        could have been done differently, and what was learned.

        Args:
            decision: The Decision that was made.
            action: The Action that was executed.
            feedback: The Feedback received.

        Returns:
            A reflection dictionary with assessment and lessons.
        """
        prompt = PromptTemplates.reflection(
            action_taken=str(action),
            outcome=str(feedback),
            situation_before=decision.situation_summary,
        )

        try:
            reflection = self.llm.generate_json(
                prompt=prompt,
                system_msg=(
                    f"You are {self.agent_name}. Reflect on your recent "
                    "decision and its outcome."
                ),
                temperature=0.5,
            )
        except (ValueError, RuntimeError) as exc:
            self.logger.warning(f"Reflection failed: {exc}")
            reflection = {
                "outcome_assessment": "Could not assess (LLM error).",
                "alternative_approach": "N/A",
                "lesson_learned": "LLM reflection call failed.",
                "would_repeat": True,
                "reflection_summary": f"Reflection error: {exc}",
            }

        # Store the full episode in episodic memory
        self.memory.record_episode(
            observation=decision.situation_summary,
            decision=decision.to_dict(),
            action=action.to_dict(),
            feedback=feedback.to_dict() if hasattr(feedback, "to_dict") else {},
            reflection=reflection,
        )

        self.logger.info(
            f"Reflection: {reflection.get('reflection_summary', 'No summary')}",
            would_repeat=reflection.get("would_repeat"),
        )

        return reflection

    def _check_uncertainty(
        self,
        situation: dict[str, Any],
        top_candidate: dict,
        confidence: float,
    ) -> dict[str, Any]:
        """Check whether the agent should act or gather more information.

        Uses the LLM to decide between acting, gathering info, or waiting
        when confidence is below the threshold.

        Args:
            situation: The situation assessment.
            top_candidate: The top-scored candidate action.
            confidence: The current confidence level.

        Returns:
            A dict with 'recommendation', 'reasoning', and 'information_needed'.
        """
        prompt = PromptTemplates.uncertainty(
            situation_summary=situation.get("situation_summary", "Unknown"),
            proposed_action=str(top_candidate),
            confidence=f"{confidence:.2f}",
        )

        try:
            result = self.llm.generate_json(
                prompt=prompt,
                system_msg=(
                    f"You are {self.agent_name}. Decide whether to act now "
                    "or gather more information."
                ),
                temperature=0.3,
            )
            return result
        except (ValueError, RuntimeError) as exc:
            self.logger.warning(f"Uncertainty check failed: {exc}")
            return {
                "recommendation": "gather_info",
                "reasoning": f"Uncertainty check LLM failed: {exc}. Defaulting to gather info.",
                "information_needed": [],
                "urgency_override": False,
            }

    def _format_working_memory(self) -> str:
        """Format working memory context as text for prompts."""
        wm = self.memory.working
        context_items = []
        for key, value in wm._context.items():
            context_items.append(f"- {key}: {value}")

        if context_items:
            return "\n".join(context_items)
        return "No additional context in working memory."

    def set_environment_ref(self, env: Any) -> None:
        """Set a reference to the environment for task information retrieval.

        Args:
            env: The environment instance.
        """
        self._env_ref = env
