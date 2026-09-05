"""Combined common-sense and persistent-memory tool."""

from __future__ import annotations

from typing import Any

from memory.knowledge_base import HouseholdKnowledge
from memory.long_term import LongTermMemory
from tools.base import BaseTool


class MemoryTool(BaseTool):
    name = "memory"
    description = "Query household common sense and learned environment memories, then store, verify, or update useful findings."
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "recall",
                    "remember",
                    "verify_memory",
                    "verify",
                    "update_memory",
                    "query_object_location",
                    "get_knowledge",
                    "get_room_objects",
                    "get_all_memories",
                    "clear_short_term",
                ],
            },
            "query": {"type": "string"},
            "content": {"type": "object"},
            "memory_type": {"type": "string"},
            "memory_id": {"type": "integer"},
            "limit": {"type": "integer"},
            "category": {"type": "string"},
            "room_name": {"type": "string"},
        },
        "required": ["action"],
    }

    def __init__(
        self,
        db_path: str = "data/memory.db",
        knowledge: HouseholdKnowledge | None = None,
        long_term: LongTermMemory | None = None,
    ) -> None:
        self.knowledge = knowledge or HouseholdKnowledge()
        self.long_term = long_term or LongTermMemory(db_path)
        self._short_term: list[dict[str, Any]] = []

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        action = parameters["action"]
        try:
            if action == "recall":
                return self._recall(str(parameters.get("query", "")), int(parameters.get("limit", 5)))
            if action == "remember":
                content = parameters.get("content")
                if not isinstance(content, dict):
                    return {"success": False, "error": "remember requires object content."}
                record = self.long_term.remember(
                    str(parameters.get("memory_type", content.get("type", "object_location"))),
                    content,
                    confidence=float(content.get("confidence", 0.6)),
                )
                self._short_term.append(record.to_dict())
                return {"success": True, "data": {"memory": record.to_dict()}}
            if action in {"verify_memory", "verify"}:
                return self._verify(parameters.get("memory_id"))
            if action == "update_memory":
                return self._update(parameters.get("memory_id"), parameters.get("content"))
            if action == "query_object_location":
                return self._object_locations(str(parameters.get("query", "")))
            if action == "get_knowledge":
                results = self.knowledge.query(str(parameters.get("query", "")), parameters.get("category"))
                return {"success": True, "data": {"results": results, "count": len(results)}}
            if action == "get_room_objects":
                room = str(parameters.get("room_name", parameters.get("query", "")))
                objects = self.knowledge.query_room_objects(room)
                return {"success": True, "data": {"room": room, "objects": objects, "count": len(objects)}}
            if action == "get_all_memories":
                records = [record.to_dict() for record in self.long_term.all(int(parameters.get("limit", 100)))]
                return {"success": True, "data": {"results": records, "count": len(records)}}
            if action == "clear_short_term":
                cleared = len(self._short_term)
                self._short_term.clear()
                return {"success": True, "data": {"cleared": cleared}}
        except (KeyError, ValueError) as error:
            return {"success": False, "error": str(error)}
        return {"success": False, "error": f"Unsupported memory action '{action}'."}

    def _recall(self, query: str, limit: int) -> dict[str, Any]:
        if not query.strip():
            return {"success": False, "error": "recall requires a non-empty query."}
        memory_results = [
            {"source": "long_term", **record.to_dict()}
            for record in self.long_term.recall(query, limit)
        ]
        knowledge_results = self.knowledge.query(query)
        results = sorted(
            memory_results + knowledge_results,
            key=lambda item: float(item.get("score", item.get("confidence", 0))),
            reverse=True,
        )[: max(1, min(limit, 50))]
        return {"success": True, "data": {"query": query, "results": results, "count": len(results)}}

    def _verify(self, memory_id: Any) -> dict[str, Any]:
        if not isinstance(memory_id, int):
            return {"success": False, "error": "verify_memory requires integer memory_id."}
        return {"success": True, "data": {"memory": self.long_term.verify(memory_id).to_dict()}}

    def _update(self, memory_id: Any, content: Any) -> dict[str, Any]:
        if not isinstance(memory_id, int) or not isinstance(content, dict):
            return {"success": False, "error": "update_memory requires memory_id and object content."}
        return {"success": True, "data": {"memory": self.long_term.update(memory_id, content).to_dict()}}

    def _object_locations(self, object_name: str) -> dict[str, Any]:
        dynamic = [
            {"source": "long_term", **record.to_dict()}
            for record in self.long_term.recall(object_name, 10)
            if record.memory_type == "object_location"
        ]
        likely_rooms = self.knowledge.query_object_likely_rooms(object_name)
        return {
            "success": True,
            "data": {
                "object": object_name,
                "memories": dynamic,
                "likely_rooms": likely_rooms,
                "count": len(dynamic),
            },
        }
