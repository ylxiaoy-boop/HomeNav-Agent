"""Shared contract for all capabilities exposed to the central agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """A plugin capability with structured input and output."""

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = {}
    tool_version: str = "1.0"

    @abstractmethod
    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Run the capability and return ``success`` plus ``data`` or ``error``."""

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
            "returns": {
                "success": "boolean",
                "data": "object when success is true",
                "error": "string when success is false",
            },
            "version": self.tool_version,
        }
