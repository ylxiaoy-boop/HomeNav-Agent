"""A reproducible 2D household environment for Agent-Tool navigation studies."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Iterable


Cell = tuple[int, int]


@dataclass(frozen=True)
class HouseholdObject:
    name: str
    room: str
    cell: Cell
    confidence: float = 0.95


class TwoDHomeEnvironment:
    """Grid world with room labels, furniture obstacles, and visible objects.

    This is intentionally a 2D discrete prototype. It validates task planning,
    tool scheduling, memory reuse, and path selection without claiming camera,
    continuous-control, or physical-robot performance.
    """

    width = 32
    height = 23
    backend = "two_d_grid"
    start_cell: Cell = (16, 11)

    room_rectangles: dict[str, tuple[int, int, int, int]] = {
        "客厅": (1, 1, 14, 11),
        "厨房": (18, 1, 30, 9),
        "卧室": (1, 13, 14, 21),
        "卫生间": (18, 12, 23, 21),
        "书房": (25, 12, 30, 21),
    }
    room_centers: dict[str, Cell] = {
        "客厅": (6, 9),
        "厨房": (22, 6),
        "卧室": (6, 18),
        "卫生间": (21, 17),
        "书房": (28, 17),
    }
    room_order = ("客厅", "厨房", "卫生间", "卧室", "书房")
    _aliases = {
        "cup": "杯子",
        "glass": "杯子",
        "mug": "杯子",
        "remote": "遥控器",
        "remote control": "遥控器",
        "phone": "手机",
        "towel": "毛巾",
        "toothbrush": "牙刷",
        "keys": "钥匙",
        "key": "钥匙",
        "book": "书",
        "kitchen": "厨房",
        "living room": "客厅",
        "bedroom": "卧室",
        "bathroom": "卫生间",
        "study": "书房",
    }

    def __init__(self, initial_cell: Cell | None = None) -> None:
        self.agent_cell = initial_cell or self.start_cell
        self.last_path: list[Cell] = [self.agent_cell]
        self._obstacles = self._build_obstacles()
        self.objects = (
            HouseholdObject("杯子", "厨房", (22, 4)),
            HouseholdObject("遥控器", "客厅", (6, 8)),
            HouseholdObject("手机", "卧室", (5, 18)),
            HouseholdObject("毛巾", "卫生间", (21, 16)),
            HouseholdObject("牙刷", "卫生间", (20, 18)),
            HouseholdObject("钥匙", "书房", (28, 16)),
            HouseholdObject("书", "书房", (27, 19)),
        )

    def reset(self, initial_cell: Cell | None = None) -> None:
        self.agent_cell = initial_cell or self.start_cell
        self.last_path = [self.agent_cell]

    def normalize_target(self, target: str) -> str:
        cleaned = target.strip().lower()
        return self._aliases.get(cleaned, target.strip())

    def _matches(self, target: str, candidate: str) -> bool:
        canonical_target = self.normalize_target(target)
        canonical_candidate = self.normalize_target(candidate)
        return (
            canonical_target == canonical_candidate
            or canonical_target in canonical_candidate
            or canonical_candidate in canonical_target
        )

    def locate_target(self, target: str) -> HouseholdObject | None:
        canonical = self.normalize_target(target)
        return next((item for item in self.objects if self._matches(canonical, item.name)), None)

    def room_for_cell(self, cell: Cell) -> str:
        x, y = cell
        for room, (left, top, right, bottom) in self.room_rectangles.items():
            if left <= x <= right and top <= y <= bottom:
                return room
        return "走廊"

    def current_view(self) -> dict[str, Any]:
        return {
            "room": self.room_for_cell(self.agent_cell),
            "position": self.position_label(self.agent_cell),
            "objects": [item.name for item in self.visible_household_objects()],
            "backend": self.backend,
            "grid_position": {"x": self.agent_cell[0], "y": self.agent_cell[1]},
        }

    def position_label(self, cell: Cell) -> str:
        return f"{self.room_for_cell(cell)}网格({cell[0]}, {cell[1]})"

    def visible_household_objects(self, target: str | None = None) -> list[HouseholdObject]:
        canonical = self.normalize_target(target) if target else None
        visible: list[HouseholdObject] = []
        for item in self.objects:
            if canonical and not self._matches(canonical, item.name):
                continue
            if self._manhattan(self.agent_cell, item.cell) <= 2:
                visible.append(item)
        return visible

    def visible_objects(self, target: str | None = None) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for index, item in enumerate(self.visible_household_objects(target)):
            x, y = item.cell
            detections.append(
                {
                    "class_name": item.name,
                    "confidence": item.confidence,
                    "bbox": [x * 20 + 10, y * 20 + 10, x * 20 + 28, y * 20 + 28],
                    "room": item.room,
                    "grid_position": {"x": x, "y": y},
                    "index": index,
                }
            )
        return detections

    def navigate(self, target: str, max_steps: int) -> dict[str, Any]:
        canonical = self.normalize_target(target)
        destination, target_object = self._destination(canonical)
        if destination is None:
            return {"success": False, "error": f"No 2D-map route for target '{target}'."}
        path = self.shortest_path(self.agent_cell, destination)
        if path is None:
            return {"success": False, "error": f"No traversable 2D-map route to '{target}'."}
        steps = len(path) - 1
        if steps > max_steps:
            return {
                "success": False,
                "error": f"2D navigation requires {steps} steps but max_steps is {max_steps}.",
            }
        self.agent_cell = destination
        self.last_path = path
        view = self.current_view()
        return {
            "success": True,
            "data": {
                "room": view["room"],
                "position": view["position"],
                "grid_position": view["grid_position"],
                "steps": steps,
                "target": target,
                "target_object": target_object.name if target_object else None,
                "path": [{"x": x, "y": y} for x, y in path],
                "detected_objects": view["objects"],
                "backend": self.backend,
            },
        }

    def shortest_path(self, start: Cell, goal: Cell) -> list[Cell] | None:
        frontier: list[tuple[int, int, Cell]] = [(0, 0, start)]
        cost: dict[Cell, int] = {start: 0}
        parent: dict[Cell, Cell | None] = {start: None}
        sequence = 0
        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                return self._reconstruct(parent, current)
            for neighbor in self._neighbors(current):
                next_cost = cost[current] + 1
                if next_cost >= cost.get(neighbor, 10**9):
                    continue
                cost[neighbor] = next_cost
                parent[neighbor] = current
                sequence += 1
                priority = next_cost + self._manhattan(neighbor, goal)
                heapq.heappush(frontier, (priority, sequence, neighbor))
        return None

    def map_snapshot(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "rooms": self.room_rectangles,
            "objects": [
                {"name": item.name, "room": item.room, "x": item.cell[0], "y": item.cell[1]}
                for item in self.objects
            ],
            "obstacles": [{"x": x, "y": y} for x, y in sorted(self._obstacles)],
            "agent": {"x": self.agent_cell[0], "y": self.agent_cell[1]},
        }

    def _destination(self, canonical: str) -> tuple[Cell | None, HouseholdObject | None]:
        if canonical in self.room_centers:
            return self.room_centers[canonical], None
        target_object = self.locate_target(canonical)
        if target_object:
            return target_object.cell, target_object
        return None, None

    def _build_obstacles(self) -> set[Cell]:
        blocks = set()
        for left, top, right, bottom in [
            (3, 3, 8, 4),   # living-room sofa
            (8, 6, 10, 7),  # coffee table
            (24, 2, 29, 3), # kitchen counter
            (24, 6, 27, 7), # kitchen island
            (8, 16, 12, 18),# bedroom bed
            (19, 14, 20, 15), # bathroom fixture
            (25, 14, 26, 18), # study desk
        ]:
            blocks.update((x, y) for x in range(left, right + 1) for y in range(top, bottom + 1))
        return blocks

    def _neighbors(self, cell: Cell) -> Iterable[Cell]:
        x, y = cell
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            nx, ny = neighbor
            if 0 <= nx < self.width and 0 <= ny < self.height and neighbor not in self._obstacles:
                yield neighbor

    @staticmethod
    def _reconstruct(parent: dict[Cell, Cell | None], current: Cell) -> list[Cell]:
        path = [current]
        while parent[current] is not None:
            current = parent[current]  # type: ignore[assignment]
            path.append(current)
        return list(reversed(path))

    @staticmethod
    def _manhattan(left: Cell, right: Cell) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])
