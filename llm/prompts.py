"""Prompt templates for the Cognition layer's LLM interactions.

Each template uses Python str.format() placeholders for dynamic insertion.
The prompts are carefully engineered to elicit structured, actionable
responses from the LLM that can be parsed by the cognition layer.

All prompts include explicit JSON-only output instructions to work reliably
with local LLMs (Qwen, Gemma, etc.) that may otherwise add thinking blocks,
markdown, or explanatory text.
"""

# ---------------------------------------------------------------------------
# Common JSON enforcement suffix — appended to every prompt
# ---------------------------------------------------------------------------
_JSON_ENFORCE = """\

REMEMBER: Output ONLY the raw JSON object. No thinking tags, no markdown, no explanation before or after the JSON. Start your response with {{ and end with }}."""

# ---------------------------------------------------------------------------
# Situation Assessment
# ---------------------------------------------------------------------------
SITUATION_ASSESSMENT_PROMPT = """\
You are an autonomous agent assessing its current situation.

## Current Observations (most recent first)
{observations}

## Working Memory Context
{working_memory}

## Recent Decisions
{recent_decisions}

## Task
Provide a concise situation assessment that:
1. Summarizes the current state of the environment in 2-3 sentences.
2. Identifies the most urgent or important items requiring attention.
3. Notes any patterns, trends, or anomalies across observations.
4. Flags any unresolved issues or risks.

Output this exact JSON structure:
{{
  "situation_summary": "Your 2-3 sentence summary here",
  "urgent_items": ["item1", "item2"],
  "patterns": ["pattern1"],
  "risks": ["risk1"],
  "overall_urgency": "low"
}}

The overall_urgency must be one of: low, medium, high, critical.
""" + _JSON_ENFORCE

# ---------------------------------------------------------------------------
# Action Generation
# ---------------------------------------------------------------------------
ACTION_GENERATION_PROMPT = """\
You are an autonomous agent generating candidate actions.

## Situation Assessment
{situation_summary}

## Urgent Items
{urgent_items}

## Available Actions
- respond: Send a message or status update.
- query_environment: Request more information from the environment.
- wait: Do nothing for now; wait for new information.
- escalate: Escalate an issue to a higher authority or human operator.
- tool_use: Use a specific tool or system command (start_task, defer_task, request_info).

## Available Tasks in Environment
{available_tasks}

## Constraints
- Tasks with unmet dependencies cannot be started.
- Higher priority tasks (5=highest) should be addressed before lower ones.
- Approaching deadlines increase urgency.
- Do not start more than one task at a time.

## Task
Generate 3-5 candidate actions the agent should consider. For each action, provide:
1. A short name in kebab-case (e.g., "start-high-priority-task")
2. A description of what the action does
3. Which action type it uses (respond, query_environment, wait, escalate, tool_use)
4. The specific parameters for that action type

Output this exact JSON structure:
{{
  "candidates": [
    {{
      "name": "action-name",
      "description": "what this action does",
      "action_type": "tool_use",
      "params": {{"tool": "start_task", "task_id": "T-001"}},
      "rationale": "why this is a good candidate"
    }}
  ]
}}
""" + _JSON_ENFORCE

# ---------------------------------------------------------------------------
# Action Evaluation
# ---------------------------------------------------------------------------
ACTION_EVALUATION_PROMPT = """\
You are an autonomous agent evaluating candidate actions.

## Situation Assessment
{situation_summary}

## Candidate Actions
{candidate_actions}

## Task
Score each candidate action on the following dimensions (0.0 to 1.0):

1. effectiveness: How effectively does this action address the situation?
2. risk: What is the level of risk? (1.0 = very risky, 0.0 = safe)
3. information_gain: How much new information will this action provide?
4. alignment: How well does this align with the agent's goals and constraints?

Compute total_score as: effectiveness * 0.4 + (1 - risk) * 0.3 + information_gain * 0.15 + alignment * 0.15

Provide your overall confidence in this evaluation (0.0 to 1.0).

Output this exact JSON structure:
{{
  "evaluations": [
    {{
      "name": "action-name",
      "effectiveness": 0.8,
      "risk": 0.2,
      "information_gain": 0.5,
      "alignment": 0.9,
      "total_score": 0.71
    }}
  ],
  "confidence": 0.75,
  "reasoning": "Brief chain-of-thought explaining the evaluation"
}}
""" + _JSON_ENFORCE

# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------
REFLECTION_PROMPT = """\
You are an autonomous agent reflecting on a completed action.

## Action Taken
{action_taken}

## Action Outcome
{outcome}

## Situation Before Action
{situation_before}

## Task
Reflect on this decision cycle:
1. Did the action achieve its intended outcome?
2. What could have been done differently?
3. What was learned that should inform future decisions?
4. Would you make the same decision again with the same information?

Output this exact JSON structure:
{{
  "outcome_assessment": "Did it work?",
  "alternative_approach": "What else could have been done",
  "lesson_learned": "Key takeaway",
  "would_repeat": true,
  "reflection_summary": "1-2 sentence summary"
}}

Note: would_repeat must be true or false (lowercase).
""" + _JSON_ENFORCE

# ---------------------------------------------------------------------------
# Uncertainty Check
# ---------------------------------------------------------------------------
UNCERTAINTY_PROMPT = """\
You are an autonomous agent deciding whether to act now or gather more information.

## Situation Assessment
{situation_summary}

## Proposed Action
{proposed_action}

## Current Confidence: {confidence}

## Task
Given the current confidence level and situation, should the agent:
- act: Proceed with the proposed action now.
- gather_info: Query the environment for more information before acting.
- wait: Wait for the situation to evolve naturally.

Consider:
- Is the confidence level sufficient for this type of action?
- Would more information significantly change the decision?
- Is there time pressure that requires immediate action?

Output this exact JSON structure:
{{
  "recommendation": "act",
  "reasoning": "Why you recommend this",
  "information_needed": ["what info would help"],
  "urgency_override": false
}}

Note: recommendation must be one of: act, gather_info, wait.
urgency_override must be true or false (lowercase).
""" + _JSON_ENFORCE


class PromptTemplates:
    """Convenience accessor for all prompt templates.

    Provides methods that format each template with the given arguments,
    handling missing keys gracefully.
    """

    @staticmethod
    def situation_assessment(**kwargs: str) -> str:
        """Format the situation assessment prompt."""
        return SITUATION_ASSESSMENT_PROMPT.format(**kwargs)

    @staticmethod
    def action_generation(**kwargs: str) -> str:
        """Format the action generation prompt."""
        return ACTION_GENERATION_PROMPT.format(**kwargs)

    @staticmethod
    def action_evaluation(**kwargs: str) -> str:
        """Format the action evaluation prompt."""
        return ACTION_EVALUATION_PROMPT.format(**kwargs)

    @staticmethod
    def reflection(**kwargs: str) -> str:
        """Format the reflection prompt."""
        return REFLECTION_PROMPT.format(**kwargs)

    @staticmethod
    def uncertainty(**kwargs: str) -> str:
        """Format the uncertainty check prompt."""
        return UNCERTAINTY_PROMPT.format(**kwargs)
