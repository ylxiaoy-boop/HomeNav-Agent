"""Unit and integration coverage for the documented Agent-Tool workflow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.parser import OutputParser
from interfaces.cli import build_agent
from memory.long_term import LongTermMemory
from models.two_d_environment import TwoDHomeEnvironment
from tools.manager import ToolManager
from tools.perception import PerceptionTool


class TestConfig:
    def __init__(self, db_path: str) -> None:
        self.values = {
            "system.max_agent_steps": 10,
            "navigation.max_steps": 100,
            "perception.confidence_threshold": 0.5,
            "memory.db_path": db_path,
            "llm.model": "mock",
            "llm.temperature": 0,
            "llm.max_tokens": 100,
            "llm.timeout_seconds": 1,
        }

    def get(self, key: str, default=None):
        return self.values.get(key, default)


class OutputParserTests(unittest.TestCase):
    def test_parses_markdown_json_and_finish_action(self) -> None:
        parser = OutputParser()
        action = parser.parse(
            "Thought: inspect the room\nAction: ```json\n"
            '{"tool_name":"perception","parameters":{"mode":"detect_objects"}}\n```'
        )
        self.assertEqual(action.tool_name, "perception")
        self.assertEqual(action.parameters["mode"], "detect_objects")

        finish = parser.parse('Action: {"type":"finish","content":"done"}')
        self.assertTrue(finish.is_finish)
        self.assertEqual(finish.finish_content, "done")


class ToolManagerTests(unittest.TestCase):
    def test_validation_and_metadata_are_standardized(self) -> None:
        manager = ToolManager()
        manager.register(PerceptionTool())

        invalid = manager.execute("perception", {})
        self.assertFalse(invalid["success"])
        self.assertIn("Missing required parameter", invalid["error"])
        self.assertEqual(invalid["metadata"]["call_id"], 1)

        valid = manager.execute("perception", {"mode": "detect_objects"})
        self.assertTrue(valid["success"])
        self.assertIsNone(valid["error"])
        self.assertEqual(valid["metadata"]["tool_name"], "perception")


class MemoryTests(unittest.TestCase):
    def test_memory_verification_increases_confidence_and_recall_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = LongTermMemory(str(Path(directory) / "memory.db"))
            record = memory.remember(
                "object_location",
                {"object": "\u676f\u5b50", "room": "\u53a8\u623f"},
            )
            verified = memory.verify(record.id)
            self.assertGreater(verified.confidence, record.confidence)

            recalled = memory.recall("\u676f\u5b50")
            self.assertEqual(recalled[0].id, record.id)
            self.assertIsNotNone(recalled[0].score)


class AgentWorkflowTests(unittest.TestCase):
    def test_mock_agent_reacts_through_memory_navigation_perception_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = build_agent(TestConfig(str(Path(directory) / "memory.db")))
            agent.llm.use_mock = True
            agent.llm.client = None

            result = agent.run("\u5e2e\u6211\u627e\u9065\u63a7\u5668")
            trace = agent.get_trace()

            self.assertIn("\u9065\u63a7\u5668", result)
            self.assertEqual(
                [entry["tool_name"] for entry in trace["tool_history"]],
                ["memory", "navigation", "perception", "memory"],
            )
            self.assertEqual(trace["current_position"]["room"], "\u5ba2\u5385")
            self.assertTrue(
                any(item["class_name"] == "\u9065\u63a7\u5668" for item in trace["found_objects"])
            )


class TwoDEnvironmentTests(unittest.TestCase):
    def test_grid_navigation_reaches_a_visible_target(self) -> None:
        environment = TwoDHomeEnvironment()
        result = environment.navigate("cup", max_steps=100)

        self.assertTrue(result["success"])
        self.assertGreater(result["data"]["steps"], 0)
        self.assertEqual(result["data"]["backend"], "two_d_grid")
        self.assertTrue(environment.visible_objects("cup"))
        self.assertEqual(environment.current_view()["room"], "\u53a8\u623f")

    def test_agent_uses_the_existing_tool_chain_on_the_2d_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = TestConfig(str(Path(directory) / "memory.db"))
            config.values["system.mode"] = "two_d"
            agent = build_agent(config)
            agent.llm.use_mock = True
            agent.llm.client = None

            result = agent.run("I am thirsty")
            trace = agent.get_trace()

            self.assertIn("\u676f\u5b50", result)
            self.assertEqual(
                [entry["tool_name"] for entry in trace["tool_history"]],
                ["memory", "navigation", "perception", "memory"],
            )
            self.assertEqual(trace["current_position"]["room"], "\u53a8\u623f")


if __name__ == "__main__":
    unittest.main()
