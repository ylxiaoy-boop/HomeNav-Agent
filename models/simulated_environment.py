"""A deterministic household world used when robot hardware is not configured."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimulatedHomeEnvironment:
    """Small shared world state for local navigation and visual perception tests."""

    rooms: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "\u53a8\u623f": {
            "position": "\u53a8\u623f\u4e2d\u592e",
            "objects": ["\u51b0\u7bb1", "\u7076\u53f0", "\u6c34\u69fd", "\u676f\u5b50", "\u7897", "\u76d8\u5b50", "\u5200", "\u7b77\u5b50"],
        },
        "\u5ba2\u5385": {
            "position": "\u5ba2\u5385\u8336\u51e0\u65c1",
            "objects": ["\u6c99\u53d1", "\u7535\u89c6", "\u9065\u63a7\u5668", "\u62b1\u6795", "\u6c34\u676f", "\u624b\u673a", "\u53f0\u706f"],
        },
        "\u5367\u5ba4": {
            "position": "\u5367\u5ba4\u5e8a\u8fb9",
            "objects": ["\u5e8a", "\u8863\u67dc", "\u53f0\u706f", "\u624b\u673a", "\u4e66", "\u6795\u5934", "\u88ab\u5b50"],
        },
        "\u536b\u751f\u95f4": {
            "position": "\u536b\u751f\u95f4\u6d17\u624b\u53f0\u524d",
            "objects": ["\u9a6c\u6876", "\u6d17\u624b\u6c60", "\u6bdb\u5dfe", "\u7259\u5237", "\u7259\u818f", "\u955c\u5b50"],
        },
        "\u4e66\u623f": {
            "position": "\u4e66\u623f\u4e66\u684c\u524d",
            "objects": ["\u4e66\u684c", "\u7535\u8111", "\u7b14", "\u672c\u5b50", "\u53f0\u706f", "\u4e66\u67b6"],
        },
    })
    current_room: str = "\u5ba2\u5385"

    _aliases: dict[str, str] = field(default_factory=lambda: {
        "kitchen": "\u53a8\u623f",
        "living room": "\u5ba2\u5385",
        "bedroom": "\u5367\u5ba4",
        "bathroom": "\u536b\u751f\u95f4",
        "study": "\u4e66\u623f",
        "cup": "\u676f\u5b50",
        "glass": "\u676f\u5b50",
        "remote": "\u9065\u63a7\u5668",
        "remote control": "\u9065\u63a7\u5668",
        "phone": "\u624b\u673a",
        "towel": "\u6bdb\u5dfe",
        "toothbrush": "\u7259\u5237",
    })

    def normalize_target(self, target: str) -> str:
        cleaned = target.strip().lower()
        return self._aliases.get(cleaned, target.strip())

    def locate_target(self, target: str) -> str | None:
        normalized = self.normalize_target(target)
        for room, details in self.rooms.items():
            if self._matches(normalized, room):
                return room
            if any(self._matches(normalized, item) for item in details["objects"]):
                return room
        return None

    def navigate(self, target: str, max_steps: int) -> dict[str, Any]:
        room = self.locate_target(target)
        if room is None:
            return {"success": False, "error": f"No known route for target '{target}'."}

        route_steps = 4 + abs(list(self.rooms).index(self.current_room) - list(self.rooms).index(room)) * 5
        if max_steps < route_steps:
            return {
                "success": False,
                "error": f"Navigation needs {route_steps} steps but max_steps is {max_steps}.",
            }

        self.current_room = room
        details = self.rooms[room]
        return {
            "success": True,
            "data": {
                "room": room,
                "position": details["position"],
                "steps": route_steps,
                "target": target,
                "detected_objects": list(details["objects"]),
                "backend": "simulated",
            },
        }

    def current_view(self) -> dict[str, Any]:
        details = self.rooms[self.current_room]
        return {
            "room": self.current_room,
            "position": details["position"],
            "objects": list(details["objects"]),
        }

    @staticmethod
    def _matches(target: str, candidate: str) -> bool:
        return target == candidate or target in candidate or candidate in target
