"""Parsing utilities for structured ReAct model responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class ActionParseError(ValueError):
    """Raised when an LLM response does not contain a usable action."""


@dataclass(frozen=True)
class ParsedAction:
    thought: str
    tool_name: str | None
    parameters: dict[str, Any]
    finish_content: str | None = None

    @property
    def is_finish(self) -> bool:
        return self.finish_content is not None


class OutputParser:
    """Parse the documented Thought/Action response contract with small repairs."""

    _thought_pattern = re.compile(
        r"(?:^|\n)\s*Thought\s*:\s*(.*?)(?=\n\s*Action\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _action_pattern = re.compile(
        r"(?:^|\n)\s*Action\s*:\s*(.+)", re.IGNORECASE | re.DOTALL
    )

    def parse(self, response: str) -> ParsedAction:
        if not isinstance(response, str) or not response.strip():
            raise ActionParseError("The model returned an empty response.")

        response = response.strip()
        thought = self._extract_thought(response)
        action_payload = self._extract_action_payload(response)
        action = self._decode_object(action_payload)

        if action.get("type") == "finish" or action.get("tool_name") == "finish":
            parameters = action.get("parameters") or {}
            content = action.get("content") or parameters.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ActionParseError("A finish action must include non-empty content.")
            return ParsedAction(
                thought=thought,
                tool_name=None,
                parameters={},
                finish_content=content.strip(),
            )

        tool_name = action.get("tool_name")
        parameters = action.get("parameters", {})
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ActionParseError("Action.tool_name must be a non-empty string.")
        if not isinstance(parameters, dict):
            raise ActionParseError("Action.parameters must be a JSON object.")
        return ParsedAction(thought=thought, tool_name=tool_name.strip(), parameters=parameters)

    def _extract_thought(self, response: str) -> str:
        match = self._thought_pattern.search(response)
        if match:
            return match.group(1).strip() or "No reasoning text was supplied."
        return "No reasoning text was supplied."

    def _extract_action_payload(self, response: str) -> str:
        match = self._action_pattern.search(response)
        if match:
            payload = match.group(1).strip()
        else:
            payload = response

        payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload).strip()
        return payload

    def _decode_object(self, payload: str) -> dict[str, Any]:
        first_brace = payload.find("{")
        if first_brace < 0:
            raise ActionParseError("Action must be a JSON object.")

        candidate = payload[first_brace:].replace("\u201c", '"').replace("\u201d", '"')
        try:
            decoder = json.JSONDecoder()
            value, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError as error:
            raise ActionParseError(f"Invalid action JSON: {error.msg}.") from error

        if not isinstance(value, dict):
            raise ActionParseError("Action JSON must be an object.")
        return value
