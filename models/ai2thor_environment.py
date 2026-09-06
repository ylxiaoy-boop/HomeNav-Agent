"""AI2-THOR household environment adapter with deterministic trajectory capture."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Callable


class AI2THORUnavailableError(RuntimeError):
    """Raised when the optional AI2-THOR runtime cannot be initialized."""


class AI2THORHomeEnvironment:
    """Expose the local environment contract on top of an AI2-THOR controller.

    Navigation uses reachable positions supplied by AI2-THOR and issues only
    RotateLeft/RotateRight/MoveAhead actions. Object metadata is used to select
    a semantic goal for evaluation; it is not presented as a YOLO measurement.
    """

    _ALIASES = {
        "cup": "cup",
        "glass": "cup",
        "\u676f\u5b50": "cup",
        "remote": "remotecontrol",
        "remotecontrol": "remotecontrol",
        "\u9065\u63a7\u5668": "remotecontrol",
        "phone": "cellphone",
        "cellphone": "cellphone",
        "\u624b\u673a": "cellphone",
        "towel": "towel",
        "\u6bdb\u5dfe": "towel",
        "toothbrush": "toothbrush",
        "\u7259\u5237": "toothbrush",
        "kitchen": "kitchen",
        "\u53a8\u623f": "kitchen",
        "livingroom": "livingroom",
        "living room": "livingroom",
        "\u5ba2\u5385": "livingroom",
        "bedroom": "bedroom",
        "\u5367\u5ba4": "bedroom",
        "bathroom": "bathroom",
        "\u536b\u751f\u95f4": "bathroom",
    }

    def __init__(
        self,
        scene: str = "FloorPlan1",
        width: int = 640,
        height: int = 480,
        grid_size: float = 0.25,
        record_path: str | Path | None = None,
        controller_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.scene = scene
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.record_path = Path(record_path) if record_path else None
        self._frames: list[Any] = []
        self._closed = False

        if controller_factory is None:
            try:
                from ai2thor.controller import Controller
            except ImportError as error:
                raise AI2THORUnavailableError(
                    "AI2-THOR is not installed. Install the project dependencies first."
                ) from error
            controller_factory = Controller

        try:
            self.controller = controller_factory(
                scene=scene,
                width=width,
                height=height,
                gridSize=grid_size,
                fieldOfView=90,
                renderDepthImage=False,
                renderInstanceSegmentation=False,
            )
        except Exception as error:
            raise AI2THORUnavailableError(
                "Unable to start AI2-THOR. AI2-THOR 5 requires a supported Linux "
                "runtime; native Windows builds are unavailable. Run this backend "
                "from Ubuntu/WSL2 with graphics support or a Linux validation host."
            ) from error

        self.last_event = self.controller.last_event
        self._record_frame(self.last_event)

    def close(self) -> None:
        """Finalize optional video recording and stop the Unity controller."""
        if self._closed:
            return
        self._closed = True
        try:
            self._write_video()
        finally:
            self.controller.stop()

    def normalize_target(self, target: str) -> str:
        key = self._normalize_text(target)
        return self._ALIASES.get(key, key)

    def navigate(self, target: str, max_steps: int) -> dict[str, Any]:
        canonical_target = self.normalize_target(target)
        candidates = self._target_candidates(canonical_target)
        if not candidates:
            return {
                "success": False,
                "error": f"No AI2-THOR object matching '{target}' was found in {self.scene}.",
            }

        reachable = self._reachable_positions()
        if not reachable:
            return {"success": False, "error": "AI2-THOR returned no reachable positions."}

        agent_position = self._agent_position()
        target_position = candidates[0].get("position", {})
        start = self._nearest_position(agent_position, reachable)
        goal = self._nearest_position(target_position, reachable)
        path = self._shortest_path(start, goal, reachable)
        if path is None:
            return {"success": False, "error": "No reachable path to the semantic target."}
        if len(path) - 1 > max_steps:
            return {
                "success": False,
                "error": f"Navigation needs {len(path) - 1} movement steps but max_steps is {max_steps}.",
            }

        executed_steps = 0
        for current, following in zip(path, path[1:]):
            self._turn_towards(current, following)
            event = self._step("MoveAhead")
            if not event.metadata.get("lastActionSuccess", False):
                return {
                    "success": False,
                    "error": event.metadata.get("errorMessage", "MoveAhead failed."),
                }
            executed_steps += 1

        self._face_position(target_position)
        view = self.current_view()
        return {
            "success": True,
            "data": {
                "room": self.scene,
                "position": view["position"],
                "steps": executed_steps,
                "target": target,
                "detected_objects": view["objects"],
                "backend": "ai2thor",
                "scene": self.scene,
                "target_object_id": candidates[0].get("objectId"),
            },
        }

    def current_view(self) -> dict[str, Any]:
        visible = [
            item for item in self.last_event.metadata.get("objects", []) if item.get("visible")
        ]
        return {
            "room": self.scene,
            "position": self._position_label(self._agent_position()),
            "objects": [item.get("objectType", "Unknown") for item in visible],
            "backend": "ai2thor",
            "scene": self.scene,
        }

    def visible_objects(self, target: str | None = None) -> list[dict[str, Any]]:
        canonical_target = self.normalize_target(target) if target else None
        detections: list[dict[str, Any]] = []
        for index, item in enumerate(self.last_event.metadata.get("objects", [])):
            if not item.get("visible"):
                continue
            object_type = str(item.get("objectType", "Unknown"))
            if canonical_target and not self._matches(canonical_target, object_type):
                continue
            detections.append(
                {
                    "class_name": object_type,
                    "confidence": 1.0,
                    "bbox": self._bbox(item, index),
                    "object_id": item.get("objectId"),
                }
            )
        return detections

    def _target_candidates(self, canonical_target: str) -> list[dict[str, Any]]:
        return [
            item
            for item in self.last_event.metadata.get("objects", [])
            if self._matches(canonical_target, str(item.get("objectType", "")))
        ]

    def _reachable_positions(self) -> list[dict[str, float]]:
        event = self._step("GetReachablePositions")
        if not event.metadata.get("lastActionSuccess", False):
            return []
        return list(event.metadata.get("actionReturn") or [])

    def _step(self, action: str, **parameters: Any) -> Any:
        self.last_event = self.controller.step(action=action, **parameters)
        self._record_frame(self.last_event)
        return self.last_event

    def _turn_towards(self, current: dict[str, float], following: dict[str, float]) -> None:
        dx = round((following["x"] - current["x"]) / self.grid_size)
        dz = round((following["z"] - current["z"]) / self.grid_size)
        desired = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}.get((dx, dz))
        if desired is None:
            raise ValueError("AI2-THOR path contains a non-cardinal move.")
        self._rotate_to(desired)

    def _face_position(self, target_position: dict[str, Any]) -> None:
        agent = self._agent_position()
        dx = float(target_position.get("x", agent["x"])) - agent["x"]
        dz = float(target_position.get("z", agent["z"])) - agent["z"]
        if abs(dx) < 0.01 and abs(dz) < 0.01:
            return
        angle = (math.degrees(math.atan2(dx, dz)) + 360) % 360
        self._rotate_to(int(round(angle / 90.0) * 90) % 360)

    def _rotate_to(self, desired: int) -> None:
        current = int(round(self.last_event.metadata["agent"]["rotation"]["y"])) % 360
        right_turns = ((desired - current) % 360) // 90
        if right_turns <= 2:
            for _ in range(right_turns):
                self._step("RotateRight")
        else:
            for _ in range(4 - right_turns):
                self._step("RotateLeft")

    def _shortest_path(
        self,
        start: dict[str, float],
        goal: dict[str, float],
        reachable: list[dict[str, float]],
    ) -> list[dict[str, float]] | None:
        positions = {self._position_key(item): item for item in reachable}
        start_key = self._position_key(start)
        goal_key = self._position_key(goal)
        if start_key not in positions:
            positions[start_key] = start
        queue: deque[tuple[int, int]] = deque([start_key])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start_key: None}
        while queue:
            current = queue.popleft()
            if current == goal_key:
                break
            for delta in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                neighbor = (current[0] + delta[0], current[1] + delta[1])
                if neighbor in positions and neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)
        if goal_key not in parent:
            return None
        keys: list[tuple[int, int]] = []
        current_key: tuple[int, int] | None = goal_key
        while current_key is not None:
            keys.append(current_key)
            current_key = parent[current_key]
        return [positions[key] for key in reversed(keys)]

    def _agent_position(self) -> dict[str, float]:
        return dict(self.last_event.metadata["agent"]["position"])

    def _nearest_position(
        self, source: dict[str, Any], candidates: list[dict[str, float]]
    ) -> dict[str, float]:
        return min(
            candidates,
            key=lambda item: (float(item["x"]) - float(source.get("x", 0))) ** 2
            + (float(item["z"]) - float(source.get("z", 0))) ** 2,
        )

    def _position_key(self, position: dict[str, float]) -> tuple[int, int]:
        return (
            round(float(position["x"]) / self.grid_size),
            round(float(position["z"]) / self.grid_size),
        )

    def _record_frame(self, event: Any) -> None:
        if self.record_path is None:
            return
        frame = getattr(event, "frame", None)
        if frame is not None:
            self._frames.append(frame.copy())

    def _write_video(self) -> None:
        if self.record_path is None or not self._frames:
            return
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required to encode AI2-THOR validation videos.") from error
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = self._frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(self.record_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open video output: {self.record_path}")
        try:
            for frame in self._frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    @staticmethod
    def _normalize_text(value: str) -> str:
        return value.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    def _matches(self, target: str, candidate: str) -> bool:
        normalized_candidate = self._normalize_text(candidate)
        return target == normalized_candidate or target in normalized_candidate

    @staticmethod
    def _position_label(position: dict[str, float]) -> str:
        return f"x={position['x']:.2f}, z={position['z']:.2f}"

    @staticmethod
    def _bbox(item: dict[str, Any], index: int) -> list[int]:
        rectangle = item.get("screenRect") or {}
        if rectangle:
            return [
                int(rectangle.get("x", 0)),
                int(rectangle.get("y", 0)),
                int(rectangle.get("x", 0) + rectangle.get("width", 0)),
                int(rectangle.get("y", 0) + rectangle.get("height", 0)),
            ]
        return [40 + index * 17, 60 + index * 11, 140 + index * 17, 160 + index * 11]
