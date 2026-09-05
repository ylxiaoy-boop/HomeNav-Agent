"""Runtime state tracked for a single HomeNav task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TaskState:
    """A serializable view of task progress for reasoning and diagnostics."""

    task_goal: str
    current_subgoal: str = ""
    current_position: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    found_objects: list[dict[str, Any]] = field(default_factory=list)
    searched_areas: list[str] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def record_tool_call(
        self,
        *,
        step: int,
        thought: str,
        tool_name: str,
        parameters: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        self.step_count = step
        entry = {
            "step": step,
            "thought": thought,
            "tool_name": tool_name,
            "parameters": parameters,
            "observation": observation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.tool_history.append(entry)
        self.observations.append(observation)
        self._apply_observation(tool_name, parameters, observation)

    def record_parse_error(self, *, step: int, raw_response: str, error: str) -> None:
        self.step_count = step
        self.observations.append(
            {
                "success": False,
                "error": error,
                "metadata": {"source": "output_parser", "raw_response": raw_response},
            }
        )

    def _apply_observation(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        if not observation.get("success"):
            return

        data = observation.get("data") or {}
        if tool_name == "navigation":
            room = data.get("room")
            if room:
                self.current_position = {
                    "room": room,
                    "position": data.get("position", room),
                }
                if room not in self.searched_areas:
                    self.searched_areas.append(room)
            self.current_subgoal = parameters.get("target", self.current_subgoal)

        if tool_name == "perception":
            objects = data.get("objects") or []
            if data.get("found"):
                objects = objects or [
                    {
                        "class_name": data.get("object"),
                        "confidence": data.get("confidence"),
                    }
                ]
            known = {item.get("class_name") for item in self.found_objects}
            for item in objects:
                if item.get("class_name") and item.get("class_name") not in known:
                    self.found_objects.append(item)
                    known.add(item.get("class_name"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def reasoning_context(self) -> dict[str, Any]:
        """Return the bounded state that is useful in the next LLM decision."""
        return {
            "task_goal": self.task_goal,
            "current_subgoal": self.current_subgoal,
            "current_position": self.current_position,
            "step_count": self.step_count,
            "found_objects": self.found_objects[-10:],
            "searched_areas": self.searched_areas,
            "recent_tool_history": self.tool_history[-5:],
        }
