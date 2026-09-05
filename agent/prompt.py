"""Prompt construction for the central decision agent."""

from __future__ import annotations

import json
from typing import Any


def build_system_prompt(tool_descriptions: list[dict[str, Any]], max_steps: int) -> str:
    tools = json.dumps(tool_descriptions, ensure_ascii=False, indent=2)
    return f"""You are HomeNav-Agent, the sole decision maker for a household service robot.

You can reason about a user goal, then invoke exactly one available tool per turn. Tools execute actions; do not invent observations. Use memory before exploring when it could help, avoid revisiting searched rooms without a reason, and recover from tool errors using their observation.

Available tools:
{tools}

Return exactly this format, with JSON on the Action line:
Thought: concise rationale for the next decision
Action: {{"tool_name": "tool_name", "parameters": {{}}}}

When the goal is complete, return:
Thought: concise completion rationale
Action: {{"type": "finish", "content": "final user-facing answer"}}

Do not call more than one tool in a turn. The task has a hard limit of {max_steps} tool-decision turns."""


def build_state_observation(state: dict[str, Any], observation: dict[str, Any]) -> str:
    return "Observation:\n" + json.dumps(
        {"tool_result": observation, "task_state": state},
        ensure_ascii=False,
        default=str,
    )
