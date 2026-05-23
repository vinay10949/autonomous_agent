"""LM Studio OpenAI-compatible client with retry logic and token tracking."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI


class LLMClient:
    """Client for interacting with a local LM Studio instance.

    Wraps the OpenAI Python client pointed at LM Studio's local endpoint.
    Supports text generation and structured JSON generation with retry logic,
    exponential backoff, and per-call token usage tracking.

    Attributes:
        client: The underlying OpenAI client instance.
        model: The model identifier to use for completions.
        temperature: Default sampling temperature.
        max_tokens: Default maximum tokens in the response.
        top_p: Default top-p (nucleus sampling) parameter.
        total_tokens_used: Cumulative token usage across all calls.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the LM Studio client from an LLM config section.

        Args:
            config: The 'llm' section of the agent configuration, containing
                'base_url', 'api_key', 'model', 'temperature', 'max_tokens', 'top_p'.
        """
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=config.get("api_key", "") or "not-needed",
        )
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.top_p = config["top_p"]
        self.total_tokens_used: int = 0
        self._call_count: int = 0
        self._json_fail_count: int = 0

    def generate(
        self,
        prompt: str,
        system_msg: str = "You are a helpful autonomous agent.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 3,
        timeout: float = 120.0,
    ) -> str:
        """Generate a text completion from the LLM.

        Args:
            prompt: The user message / prompt text.
            system_msg: The system message providing context and instructions.
            temperature: Override default temperature for this call.
            max_tokens: Override default max_tokens for this call.
            retries: Number of retry attempts on failure.
            timeout: Request timeout in seconds.

        Returns:
            The generated text content as a string.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        return self._call(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            timeout=timeout,
        )

    def generate_json(
        self,
        prompt: str,
        system_msg: str = "You are a helpful autonomous agent. Respond in JSON format.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 3,
        timeout: float = 120.0,
    ) -> dict:
        """Generate a structured JSON response from the LLM.

        The system message is augmented with instructions to respond in JSON.
        The response is parsed and returned as a dictionary. If parsing fails,
        a fallback extraction attempt is made.  On failure, the call is
        retried with an assistant pre-fill that starts the JSON object.

        Args:
            prompt: The user message / prompt text.
            system_msg: Base system message (will be extended with JSON instructions).
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            retries: Number of retry attempts on failure.
            timeout: Request timeout in seconds.

        Returns:
            A parsed dictionary from the LLM's JSON response.

        Raises:
            RuntimeError: If all retry attempts fail.
            ValueError: If the response cannot be parsed as JSON.
        """
        augmented_system = (
            system_msg
            + "\n\nCRITICAL INSTRUCTION: You MUST respond with ONLY valid JSON. "
            "Do NOT include any thinking, reasoning, explanation, or markdown. "
            "Do NOT wrap the JSON in code fences. "
            "Output ONLY the raw JSON object starting with { and ending with }. "
            "Use double quotes for all keys and string values. "
            "Use true/false (lowercase) for booleans, not True/False."
        )
        messages = [
            {"role": "system", "content": augmented_system},
            {"role": "user", "content": prompt},
        ]

        # First attempt: normal call
        try:
            raw = self._call(
                messages=messages,
                temperature=temperature or 0.3,
                max_tokens=max_tokens,
                retries=retries,
                timeout=timeout,
            )
            return self._parse_json_response(raw)
        except ValueError:
            pass  # Fall through to retry with pre-fill

        self._json_fail_count += 1

        # Second attempt: add assistant pre-fill to force JSON start
        messages_with_prefill = list(messages) + [
            {"role": "assistant", "content": "{"},
        ]

        try:
            raw = self._call(
                messages=messages_with_prefill,
                temperature=max(0.1, (temperature or 0.3) - 0.1),
                max_tokens=max_tokens,
                retries=max(1, retries - 1),
                timeout=timeout,
            )
            # Pre-fill added "{", so prepend it if the model didn't include it
            if not raw.strip().startswith("{"):
                raw = "{" + raw
            return self._parse_json_response(raw)
        except ValueError:
            pass  # Fall through to third attempt

        # Third attempt: explicit re-prompt
        explicit_messages = [
            {"role": "system", "content": augmented_system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "I will respond with only valid JSON:"},
            {"role": "user", "content": "Now output the JSON object. Start with {"},
        ]

        try:
            raw = self._call(
                messages=explicit_messages,
                temperature=0.1,
                max_tokens=max_tokens,
                retries=1,
                timeout=timeout,
            )
            return self._parse_json_response(raw)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse LLM response as JSON after multiple attempts. "
                f"Last raw response: {raw[:500]}"
            ) from exc

    def _call(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 3,
        timeout: float = 120.0,
    ) -> str:
        """Execute a chat completion call with retry logic.

        Args:
            messages: The message list for the chat completion.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            retries: Number of retries with exponential backoff.
            timeout: Request timeout in seconds.

        Returns:
            The text content of the assistant's response.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_exception: Exception | None = None

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    top_p=self.top_p,
                    timeout=timeout,
                )

                # Track token usage
                if response.usage:
                    self.total_tokens_used += response.usage.total_tokens
                self._call_count += 1

                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("LLM returned empty content")
                return content.strip()

            except Exception as exc:
                last_exception = exc
                if attempt < retries - 1:
                    backoff = 2 ** attempt
                    time.sleep(backoff)

        raise RuntimeError(
            f"LLM call failed after {retries} attempts: {last_exception}"
        )

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """Attempt to parse the LLM response as JSON using multiple strategies.

        Strategy order:
        1. Direct JSON parse.
        2. Strip <think/> blocks (common with Qwen models).
        3. Extract from markdown code fences.
        4. Find outermost balanced braces with brace-counting.
        5. Fix common LLM JSON mistakes (trailing commas, True/False, single quotes, comments).

        Args:
            raw: The raw text response from the LLM.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            ValueError: If JSON cannot be extracted from the response.
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Strip <think/> blocks and similar reasoning tags
        cleaned = re.sub(r"<think[^>]*>.*?</think\s*>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"<thinking[^>]*>.*?</thinking\s*>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<reasoning[^>]*>.*?</reasoning\s*>", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()

        if cleaned:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Extract from markdown code fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if fence_match:
            fence_content = fence_match.group(1).strip()
            try:
                return json.loads(fence_content)
            except json.JSONDecodeError:
                # Try fixing common issues in fenced content
                fixed = _fix_common_json_errors(fence_content)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        # Strategy 4: Find outermost balanced braces using brace counting
        json_str = _extract_balanced_json(cleaned if cleaned else raw)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try fixing common issues
                fixed = _fix_common_json_errors(json_str)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        # Strategy 5: Last resort — try fixing the entire cleaned response
        fixed_full = _fix_common_json_errors(cleaned if cleaned else raw)
        try:
            return json.loads(fixed_full)
        except json.JSONDecodeError:
            pass

        raise ValueError(f"Could not parse LLM response as JSON. Raw: {raw[:500]}")

    @property
    def call_count(self) -> int:
        """Number of successful LLM calls made."""
        return self._call_count

    @property
    def json_fail_count(self) -> int:
        """Number of times JSON parsing required retry or pre-fill."""
        return self._json_fail_count


def _extract_balanced_json(text: str) -> str | None:
    """Extract the outermost balanced JSON object from text using brace counting.

    This is more robust than simple find/rfind because it handles nested
    braces and skips braces inside strings.

    Args:
        text: The text to search for a JSON object.

    Returns:
        The extracted JSON string, or None if no balanced object found.
    """
    # Find the first '{'
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    i = start

    while i < len(text):
        char = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if char == "\\" and in_string:
            escape_next = True
            i += 1
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            i += 1
            continue

        if in_string:
            i += 1
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

        i += 1

    # No balanced brace found — fall back to rfind
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]

    return None


def _fix_common_json_errors(text: str) -> str:
    """Fix common JSON errors produced by LLMs.

    Handles:
    - Python-style True/False → true/false
    - Python-style None → null
    - Trailing commas before } or ]
    - Single-quoted strings → double-quoted
    - JavaScript-style comments (// and /* */)
    - Unquoted keys

    Args:
        text: The potentially broken JSON string.

    Returns:
        A hopefully-fixed JSON string.
    """
    # Remove JavaScript-style comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Fix Python-style booleans and None
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    # Fix trailing commas before closing braces/brackets
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # Fix single-quoted strings in simple cases
    # After colon: 'value' → "value"
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    # Inside arrays: 'value' → "value"
    text = re.sub(r"\[\s*'([^']*)'", r'["\1"', text)
    text = re.sub(r",\s*'([^']*)'", r', "\1"', text)

    return text
