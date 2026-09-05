"""High-level semantic navigation tool."""

from __future__ import annotations

from typing import Any

from models.vlfm_model import VLFMModel
from tools.base import BaseTool


class NavigationTool(BaseTool):
    name = "navigation"
    description = "Navigate to a room or semantic target, explore a target area, or report the current position."
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["navigate_to", "explore", "get_position"],
                "description": "Defaults to navigate_to.",
            },
            "target": {"type": "string", "description": "Room or object to navigate toward."},
            "max_steps": {"type": "integer", "description": "Maximum low-level movement steps."},
        },
    }

    def __init__(self, navigator: VLFMModel | None = None, default_max_steps: int = 100) -> None:
        self.navigator = navigator or VLFMModel()
        self.default_max_steps = default_max_steps

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        action = parameters.get("action", "navigate_to")
        if action == "get_position":
            return self.navigator.get_position()

        target = str(parameters.get("target", "")).strip()
        if not target:
            return {"success": False, "error": "Navigation requires a non-empty target."}

        result = self.navigator.navigate_to(
            target=target,
            max_steps=int(parameters.get("max_steps", self.default_max_steps)),
        )
        if result.get("success") and action == "explore":
            result["data"]["explored"] = True
        return result
