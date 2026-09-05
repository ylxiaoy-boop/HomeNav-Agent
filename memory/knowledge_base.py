"""Static household common-sense knowledge with lightweight alias matching."""

from __future__ import annotations

from typing import Any


class HouseholdKnowledge:
    """Predefined object, room, and need relationships for planning assistance."""

    def __init__(self) -> None:
        self._aliases = {
            "cup": "\u676f\u5b50",
            "glass": "\u676f\u5b50",
            "remote": "\u9065\u63a7\u5668",
            "remote control": "\u9065\u63a7\u5668",
            "phone": "\u624b\u673a",
            "towel": "\u6bdb\u5dfe",
            "toothbrush": "\u7259\u5237",
            "thirsty": "\u53e3\u6e34",
            "hungry": "\u9965",
            "cold": "\u51b7",
            "kitchen": "\u53a8\u623f",
            "living room": "\u5ba2\u5385",
            "bedroom": "\u5367\u5ba4",
            "bathroom": "\u536b\u751f\u95f4",
        }
        self._records: list[dict[str, Any]] = [
            {
                "category": "object_location",
                "key": "\u676f\u5b50",
                "content": {"object": "\u676f\u5b50", "likely_rooms": ["\u53a8\u623f", "\u5ba2\u5385"], "purpose": "\u996e\u6c34"},
            },
            {
                "category": "object_location",
                "key": "\u9065\u63a7\u5668",
                "content": {"object": "\u9065\u63a7\u5668", "likely_rooms": ["\u5ba2\u5385"], "purpose": "\u63a7\u5236\u7535\u89c6"},
            },
            {
                "category": "object_location",
                "key": "\u6bdb\u5dfe",
                "content": {"object": "\u6bdb\u5dfe", "likely_rooms": ["\u536b\u751f\u95f4", "\u5367\u5ba4"], "purpose": "\u6e05\u6d01"},
            },
            {
                "category": "object_location",
                "key": "\u7259\u5237",
                "content": {"object": "\u7259\u5237", "likely_rooms": ["\u536b\u751f\u95f4"], "purpose": "\u5237\u7259"},
            },
            {
                "category": "object_location",
                "key": "\u624b\u673a",
                "content": {"object": "\u624b\u673a", "likely_rooms": ["\u5ba2\u5385", "\u5367\u5ba4"], "purpose": "\u901a\u4fe1"},
            },
            {
                "category": "need_object",
                "key": "\u53e3\u6e34",
                "content": {"need": "\u53e3\u6e34", "objects": ["\u676f\u5b50", "\u6c34"], "likely_room": "\u53a8\u623f"},
            },
            {
                "category": "need_object",
                "key": "\u9965",
                "content": {"need": "\u9965", "objects": ["\u98df\u7269", "\u7897"], "likely_room": "\u53a8\u623f"},
            },
            {
                "category": "need_object",
                "key": "\u51b7",
                "content": {"need": "\u51b7", "objects": ["\u8863\u670d", "\u88ab\u5b50"], "likely_room": "\u5367\u5ba4"},
            },
            {
                "category": "room_info",
                "key": "\u53a8\u623f",
                "content": {"room": "\u53a8\u623f", "objects": ["\u51b0\u7bb1", "\u7076\u53f0", "\u676f\u5b50", "\u7897", "\u76d8\u5b50"]},
            },
            {
                "category": "room_info",
                "key": "\u5ba2\u5385",
                "content": {"room": "\u5ba2\u5385", "objects": ["\u6c99\u53d1", "\u7535\u89c6", "\u9065\u63a7\u5668", "\u624b\u673a"]},
            },
            {
                "category": "room_info",
                "key": "\u536b\u751f\u95f4",
                "content": {"room": "\u536b\u751f\u95f4", "objects": ["\u6bdb\u5dfe", "\u7259\u5237", "\u7259\u818f"]},
            },
        ]

    def normalize(self, query: str) -> str:
        cleaned = query.strip().lower()
        return self._aliases.get(cleaned, query.strip())

    def query(self, query: str, category: str | None = None) -> list[dict[str, Any]]:
        normalized = self.normalize(query)
        results: list[dict[str, Any]] = []
        for record in self._records:
            if category and record["category"] != category:
                continue
            if self._matches(normalized, record["key"]) or normalized in str(record["content"]):
                results.append(
                    {
                        "source": "knowledge",
                        "category": record["category"],
                        "content": record["content"],
                        "confidence": 0.8,
                        "score": 0.8,
                    }
                )
        return results

    def query_object_likely_rooms(self, object_name: str) -> list[str]:
        results = self.query(object_name, "object_location")
        return results[0]["content"].get("likely_rooms", []) if results else []

    def query_needed_objects(self, need: str) -> list[str]:
        results = self.query(need, "need_object")
        return results[0]["content"].get("objects", []) if results else []

    def query_room_objects(self, room_name: str) -> list[str]:
        results = self.query(room_name, "room_info")
        return results[0]["content"].get("objects", []) if results else []

    @staticmethod
    def _matches(query: str, candidate: str) -> bool:
        return query == candidate or query in candidate or candidate in query
