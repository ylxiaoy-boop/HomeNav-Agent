"""Plugin registry, validation, execution monitoring, and result normalization."""

from __future__ import annotations

import logging
import time
from typing import Any

from tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolManager:
    def __init__(self) -> None:
        self._registry: dict[str, BaseTool] = {}
        self._call_sequence = 0

    def register(self, tool: BaseTool, *, replace: bool = False) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError("Only BaseTool implementations can be registered.")
        if not tool.name or not tool.name.replace("_", "").isalnum():
            raise ValueError("Tool names must be non-empty lowercase identifiers.")
        if tool.name in self._registry and not replace:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._registry[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def unregister(self, tool_name: str) -> bool:
        return self._registry.pop(tool_name, None) is not None

    def get_tool(self, tool_name: str) -> BaseTool | None:
        return self._registry.get(tool_name)

    def get_all_descriptions(self) -> list[dict[str, Any]]:
        return [tool.get_metadata() for tool in self._registry.values()]

    def get_tool_names(self) -> list[str]:
        return list(self._registry)

    def execute(self, tool_name: str, parameters: dict[str, Any] | None) -> dict[str, Any]:
        self._call_sequence += 1
        call_id = self._call_sequence
        start = time.perf_counter()

        tool = self._registry.get(tool_name)
        if tool is None:
            return self._error(
                f"Unknown tool '{tool_name}'. Available tools: {', '.join(self.get_tool_names())}.",
                tool_name=tool_name,
                call_id=call_id,
                start=start,
            )

        if not isinstance(parameters, dict):
            return self._error(
                "Tool parameters must be a JSON object.",
                tool_name=tool_name,
                call_id=call_id,
                start=start,
                tool=tool,
            )

        validation_error = self._validate_parameters(tool.parameters_schema, parameters)
        if validation_error:
            return self._error(
                validation_error,
                tool_name=tool_name,
                call_id=call_id,
                start=start,
                tool=tool,
            )

        try:
            result = tool.execute(parameters)
        except Exception as error:  # Tool failures are observations, not agent-loop failures.
            logger.exception("Tool '%s' failed", tool_name)
            return self._error(
                str(error),
                tool_name=tool_name,
                call_id=call_id,
                start=start,
                tool=tool,
            )

        if not isinstance(result, dict):
            return self._error(
                "Tool returned a non-object result.",
                tool_name=tool_name,
                call_id=call_id,
                start=start,
                tool=tool,
            )

        success = bool(result.get("success"))
        normalized: dict[str, Any] = {
            "success": success,
            "data": result.get("data", {}) if success else None,
            "error": None if success else str(result.get("error") or "Tool execution failed."),
            "metadata": {
                "tool_name": tool_name,
                "tool_version": tool.tool_version,
                "call_id": call_id,
                "execution_time": round(time.perf_counter() - start, 4),
            },
        }
        extra_metadata = result.get("metadata")
        if isinstance(extra_metadata, dict):
            normalized["metadata"].update(extra_metadata)
        return normalized

    @staticmethod
    def _validate_parameters(schema: dict[str, Any], parameters: dict[str, Any]) -> str | None:
        if schema.get("type") not in (None, "object"):
            return "Tool schema root must be an object."

        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in parameters or parameters[required] is None:
                return f"Missing required parameter '{required}'."

        for name, value in parameters.items():
            definition = properties.get(name)
            if not definition:
                continue
            expected_type = definition.get("type")
            if expected_type and not ToolManager._matches_type(value, expected_type):
                return f"Parameter '{name}' must be of type {expected_type}."
            choices = definition.get("enum")
            if choices is not None and value not in choices:
                return f"Parameter '{name}' must be one of: {', '.join(map(str, choices))}."
        return None

    @staticmethod
    def _matches_type(value: Any, expected_type: str) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "object": dict,
            "array": list,
            "boolean": bool,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        if expected_type in {"integer", "number"} and isinstance(value, bool):
            return False
        return isinstance(value, expected)

    @staticmethod
    def _error(
        error: str,
        *,
        tool_name: str,
        call_id: int,
        start: float,
        tool: BaseTool | None = None,
    ) -> dict[str, Any]:
        return {
            "success": False,
            "data": None,
            "error": error,
            "metadata": {
                "tool_name": tool_name,
                "tool_version": tool.tool_version if tool else None,
                "call_id": call_id,
                "execution_time": round(time.perf_counter() - start, 4),
            },
        }
