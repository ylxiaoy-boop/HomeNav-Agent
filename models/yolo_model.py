"""Object-detector adapter with deterministic simulated detections by default."""

from __future__ import annotations

from typing import Any

from typing import Protocol

from models.simulated_environment import SimulatedHomeEnvironment


class PerceptionEnvironment(Protocol):
    def current_view(self) -> dict[str, Any]: ...

    def normalize_target(self, target: str) -> str: ...

    def _matches(self, target: str, candidate: str) -> bool: ...


class YOLOModel:
    """Provide the output shape of a detector without binding tools to Ultralytics."""

    def __init__(self, environment: PerceptionEnvironment | None = None) -> None:
        self.environment = environment or SimulatedHomeEnvironment()

    def detect(self, target: str | None = None, threshold: float = 0.5) -> dict[str, Any]:
        view = self.environment.current_view()
        visible_objects = getattr(self.environment, "visible_objects", None)
        if callable(visible_objects):
            objects = visible_objects(target=target)
            objects = [item for item in objects if item["confidence"] >= threshold]
        else:
            objects = [
                {
                    "class_name": name,
                    "confidence": round(max(threshold, 0.96 - index * 0.03), 2),
                    "bbox": [40 + index * 17, 60 + index * 11, 140 + index * 17, 160 + index * 11],
                }
                for index, name in enumerate(view["objects"])
            ]
        if target:
            normalized = self.environment.normalize_target(target)
            objects = [
                item
                for item in objects
                if self.environment._matches(normalized, item["class_name"])
            ]
        return {
            "room": view["room"],
            "position": view["position"],
            "objects": objects,
            "backend": view.get("backend", "simulated"),
        }
