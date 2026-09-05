"""The ReAct decision loop for HomeNav-Agent."""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.parser import ActionParseError, OutputParser, ParsedAction
from agent.prompt import build_state_observation, build_system_prompt
from agent.task_state import TaskState
from models.llm_client import LLMClient
from tools.manager import ToolManager

logger = logging.getLogger(__name__)


class CentralAgent:
    """A single decision maker that schedules independent tools through ReAct."""

    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager, config: Any):
        self.llm = llm_client
        self.tools = tool_manager
        self.config = config
        self.max_steps = int(config.get("system.max_agent_steps", 20))
        self.parser = OutputParser()
        self.conversation: list[dict[str, str]] = []
        self.state: TaskState | None = None
        self._last_model_response = ""

    def run(self, user_input: str) -> str:
        """Execute one user task until the model finishes or the step budget expires."""
        if not isinstance(user_input, str) or not user_input.strip():
            return "Please provide a household service task."

        self._initialize_task(user_input.strip())
        assert self.state is not None

        for step in range(1, self.max_steps + 1):
            try:
                action = self._think()
            except ActionParseError as error:
                self._record_parse_error(step, str(error))
                continue
            except Exception as error:  # Keep an unavailable model from crashing the interface.
                logger.exception("LLM reasoning failed")
                return f"Unable to reason about this task: {error}"

            if action.is_finish:
                self.state.step_count = step
                return action.finish_content or "Task completed."

            assert action.tool_name is not None
            observation = self._act(action.tool_name, action.parameters)
            self.state.record_tool_call(
                step=step,
                thought=action.thought,
                tool_name=action.tool_name,
                parameters=action.parameters,
                observation=observation,
            )
            self._observe(action, observation)

        return f"Task stopped after reaching the {self.max_steps}-step safety limit."

    def get_trace(self) -> dict[str, Any]:
        """Return a serializable trace for an API caller or a debugger."""
        return self.state.to_dict() if self.state else {}

    def _initialize_task(self, user_input: str) -> None:
        self.state = TaskState(task_goal=user_input)
        self.conversation = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.tools.get_all_descriptions(), self.max_steps
                ),
            },
            {"role": "user", "content": user_input},
        ]

    def _think(self) -> ParsedAction:
        self._last_model_response = self.llm.chat(self.conversation)
        return self.parser.parse(self._last_model_response)

    def _act(self, tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return self.tools.execute(tool_name, parameters)

    def _observe(self, action: ParsedAction, observation: dict[str, Any]) -> None:
        assert self.state is not None
        self.conversation.append(
            {
                "role": "assistant",
                "content": "Thought: "
                + action.thought
                + "\nAction: "
                + json.dumps(
                    {"tool_name": action.tool_name, "parameters": action.parameters},
                    ensure_ascii=False,
                ),
            }
        )
        self.conversation.append(
            {
                "role": "user",
                "content": build_state_observation(
                    self.state.reasoning_context(), observation
                ),
            }
        )

    def _record_parse_error(self, step: int, error: str) -> None:
        assert self.state is not None
        self.state.record_parse_error(
            step=step, raw_response=self._last_model_response, error=error
        )
        observation = {
            "success": False,
            "error": error,
            "metadata": {"source": "output_parser"},
        }
        self.conversation.append(
            {
                "role": "user",
                "content": build_state_observation(
                    self.state.reasoning_context(), observation
                ),
            }
        )
