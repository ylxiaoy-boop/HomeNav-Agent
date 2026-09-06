"""LLM client with a deterministic local planner when no provider is configured."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from config.settings import config as default_config


class LLMClient:
    def __init__(self, config: Any = default_config) -> None:
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = config.get("llm.model", "gpt-4o-mini")
        self.temperature = config.get("llm.temperature", 0)
        self.max_tokens = config.get("llm.max_tokens", 700)
        self.timeout_seconds = config.get("llm.timeout_seconds", 30)
        self.client: Any | None = None
        self.use_mock = not bool(self.api_key)

        if self.api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
                self.use_mock = False
            except ImportError:
                self.use_mock = True

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        if self.use_mock:
            return self._mock_chat(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("LLM returned an empty message.")
            return content
        except Exception:
            # Tool execution remains usable when a transient provider error occurs.
            return self._mock_chat(messages)

    def _mock_chat(self, messages: list[dict[str, str]]) -> str:
        task = self._original_task(messages)
        target = self._infer_target(task)
        actions = self._actions(messages)
        last_action = actions[-1] if actions else None
        observation = self._latest_observation(messages)

        if last_action is None:
            return self._response(
                "I should query household knowledge and prior experience before searching.",
                "memory",
                {"action": "recall", "query": target},
            )

        tool_name = last_action.get("tool_name")
        if tool_name == "memory" and last_action.get("parameters", {}).get("action") == "recall":
            room = target if self._has_learned_location(observation) else (self._preferred_room(observation) or target)
            return self._response(
                "The memory result provides a likely area, so I will navigate there.",
                "navigation",
                {"action": "navigate_to", "target": room},
            )

        if tool_name == "navigation":
            return self._response(
                "I have reached the candidate area and should visually verify the target.",
                "perception",
                {"mode": "find_object", "target": target},
            )

        if tool_name == "perception":
            result = self._tool_result(observation)
            data = result.get("data") or {}
            if data.get("found"):
                return self._response(
                    "The target is visible. I will store this confirmed location for the next task.",
                    "memory",
                    {
                        "action": "remember",
                        "memory_type": "object_location",
                        "content": {
                            "object": data.get("object", target),
                            "room": data.get("room"),
                            "position": data.get("position"),
                            "confidence": data.get("confidence", 0.6),
                        },
                    },
                )
            return self._response(
                "The target was not visible in the first candidate area. I will use semantic navigation to inspect the target location.",
                "navigation",
                {"action": "navigate_to", "target": target},
            )

        if tool_name == "memory" and last_action.get("parameters", {}).get("action") == "remember":
            result = self._tool_result(observation)
            content = (last_action.get("parameters", {}) or {}).get("content", {})
            room = content.get("room")
            object_name = content.get("object", target)
            if result.get("success"):
                return self._finish(f"Found {object_name} in {room}; the location was saved to memory.")
            return self._finish(f"Found {object_name} in {room}.")

        return self._finish("The requested task is complete.")

    @staticmethod
    def _response(thought: str, tool_name: str, parameters: dict[str, Any]) -> str:
        return "Thought: " + thought + "\nAction: " + json.dumps(
            {"tool_name": tool_name, "parameters": parameters}, ensure_ascii=False
        )

    @staticmethod
    def _finish(content: str) -> str:
        return "Thought: The goal has been completed.\nAction: " + json.dumps(
            {"type": "finish", "content": content}, ensure_ascii=False
        )

    @staticmethod
    def _original_task(messages: list[dict[str, str]]) -> str:
        for message in messages:
            content = message.get("content", "")
            if message.get("role") == "user" and not content.startswith("Observation:"):
                return content
        return ""

    @staticmethod
    def _actions(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            match = re.search(r"Action:\s*(\{.*\})", message.get("content", ""), re.DOTALL)
            if not match:
                continue
            try:
                actions.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue
        return actions

    @staticmethod
    def _latest_observation(messages: list[dict[str, str]]) -> dict[str, Any]:
        for message in reversed(messages):
            content = message.get("content", "")
            if message.get("role") == "user" and content.startswith("Observation:"):
                try:
                    return json.loads(content.split("\n", 1)[1])
                except (IndexError, json.JSONDecodeError):
                    return {}
        return {}

    @staticmethod
    def _tool_result(observation: dict[str, Any]) -> dict[str, Any]:
        result = observation.get("tool_result", {})
        return result if isinstance(result, dict) else {}

    def _preferred_room(self, observation: dict[str, Any]) -> str | None:
        result = self._tool_result(observation)
        for item in (result.get("data") or {}).get("results", []):
            content = item.get("content", {}) if isinstance(item, dict) else {}
            rooms = content.get("likely_rooms", []) if isinstance(content, dict) else []
            if rooms:
                return str(rooms[0])
            room = content.get("likely_room") if isinstance(content, dict) else None
            if room:
                return str(room)
            room = content.get("room") if isinstance(content, dict) else None
            if room:
                return str(room)
        return None

    @staticmethod
    def _has_learned_location(observation: dict[str, Any]) -> bool:
        result = LLMClient._tool_result(observation)
        for item in (result.get("data") or {}).get("results", []):
            if not isinstance(item, dict) or item.get("source") != "long_term":
                continue
            content = item.get("content")
            if isinstance(content, dict) and content.get("room"):
                return True
        return False

    @staticmethod
    def _infer_target(task: str) -> str:
        lowered = task.lower()
        candidates = {
            "\u676f\u5b50": ["\u676f\u5b50", "cup", "glass"],
            "\u9065\u63a7\u5668": ["\u9065\u63a7\u5668", "remote"],
            "\u624b\u673a": ["\u624b\u673a", "phone"],
            "\u6bdb\u5dfe": ["\u6bdb\u5dfe", "towel"],
            "\u7259\u5237": ["\u7259\u5237", "toothbrush"],
            "\u53a8\u623f": ["\u53a8\u623f", "kitchen"],
            "\u5ba2\u5385": ["\u5ba2\u5385", "living room"],
            "\u536b\u751f\u95f4": ["\u536b\u751f\u95f4", "bathroom"],
            "\u5367\u5ba4": ["\u5367\u5ba4", "bedroom"],
            "\u94a5\u5319": ["\u94a5\u5319", "keys", "key"],
            "\u4e66": ["\u4e66", "book"],
        }
        for target, words in candidates.items():
            if any(word in lowered for word in words):
                return target
        if any(word in lowered for word in ["\u6e34", "thirsty"]):
            return "\u676f\u5b50"
        if any(word in lowered for word in ["\u770b\u7535\u89c6", "watch tv", "watch television"]):
            return "\u9065\u63a7\u5668"
        if any(word in lowered for word in ["\u5237\u7259", "brush teeth", "brush my teeth"]):
            return "\u7259\u5237"
        return "\u676f\u5b50"
