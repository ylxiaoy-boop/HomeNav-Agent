"""Visual perception tool backed by a detector adapter."""

from __future__ import annotations

from typing import Any

from models.yolo_model import YOLOModel
from tools.base import BaseTool


class PerceptionTool(BaseTool):
    name = "perception"
    description = "Detect objects in the current view, find a target object, or scan the current room."
    parameters_schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["detect_objects", "detect_all", "find_object", "look_around"],
            },
            "target": {"type": "string", "description": "Required by find_object."},
            "confidence_threshold": {"type": "number"},
        },
        "required": ["mode"],
    }

    def __init__(self, detector: YOLOModel | None = None, confidence_threshold: float = 0.5) -> None:
        self.detector = detector or YOLOModel()
        self.confidence_threshold = confidence_threshold

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        mode = parameters["mode"]
        threshold = float(parameters.get("confidence_threshold", self.confidence_threshold))

        if mode in {"detect_objects", "detect_all", "look_around"}:
            detection = self.detector.detect(threshold=threshold)
            detection["mode"] = "look_around" if mode == "look_around" else "detect_objects"
            detection["count"] = len(detection["objects"])
            return {"success": True, "data": detection}

        target = str(parameters.get("target", "")).strip()
        if not target:
            return {"success": False, "error": "find_object requires a non-empty target."}

        detection = self.detector.detect(target=target, threshold=threshold)
        objects = detection["objects"]
        data = {
            "found": bool(objects),
            "object": objects[0]["class_name"] if objects else target,
            "confidence": objects[0]["confidence"] if objects else None,
            "objects": objects,
            "room": detection["room"],
            "position": detection["position"],
            "mode": "find_object",
            "backend": detection["backend"],
        }
        if not objects:
            data["reason"] = "Target is not visible in the current room."
        return {"success": True, "data": data}
