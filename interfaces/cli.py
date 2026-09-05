"""Terminal interface for running household-service tasks."""

from __future__ import annotations

import argparse
import json

from agent.central_agent import CentralAgent
from config.settings import Config, config as default_config
from models.ai2thor_environment import AI2THORHomeEnvironment
from models.llm_client import LLMClient
from models.simulated_environment import SimulatedHomeEnvironment
from models.two_d_environment import TwoDHomeEnvironment
from models.vlfm_model import VLFMModel
from models.yolo_model import YOLOModel
from tools.manager import ToolManager
from tools.memory import MemoryTool
from tools.navigation import NavigationTool
from tools.perception import PerceptionTool


def build_agent(app_config: Config = default_config) -> CentralAgent:
    """Wire the default tools around one shared environment state."""
    mode = str(app_config.get("system.mode", "simulated")).lower()
    if mode == "ai2thor":
        environment = AI2THORHomeEnvironment(
            scene=str(app_config.get("ai2thor.scene", "FloorPlan1")),
            width=int(app_config.get("ai2thor.width", 640)),
            height=int(app_config.get("ai2thor.height", 480)),
            grid_size=float(app_config.get("ai2thor.grid_size", 0.25)),
            record_path=app_config.get("ai2thor.record_path"),
        )
    elif mode in {"two_d", "2d", "grid"}:
        environment = TwoDHomeEnvironment()
    else:
        environment = SimulatedHomeEnvironment()
    manager = ToolManager()
    manager.register(
        NavigationTool(
            navigator=VLFMModel(environment),
            default_max_steps=int(app_config.get("navigation.max_steps", 100)),
        )
    )
    manager.register(
        PerceptionTool(
            detector=YOLOModel(environment),
            confidence_threshold=float(app_config.get("perception.confidence_threshold", 0.5)),
        )
    )
    manager.register(MemoryTool(db_path=str(app_config.get("memory.db_path", "data/memory.db"))))
    agent = CentralAgent(LLMClient(app_config), manager, app_config)
    agent.environment = environment
    return agent


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HomeNav-Agent household navigation assistant")
    parser.add_argument("task", nargs="*", help="Task to execute once, then exit.")
    parser.add_argument("--trace", action="store_true", help="Print the final task trace as JSON.")
    arguments = parser.parse_args(argv)
    agent = build_agent()

    try:
        if arguments.task:
            _run_task(agent, " ".join(arguments.task), arguments.trace)
            return 0

        print("HomeNav-Agent interactive mode. Type 'exit' to quit.")
        while True:
            try:
                task = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if task.lower() in {"exit", "quit", "q"}:
                return 0
            if task:
                _run_task(agent, task, arguments.trace)
    finally:
        closer = getattr(getattr(agent, "environment", None), "close", None)
        if callable(closer):
            closer()


def _run_task(agent: CentralAgent, task: str, show_trace: bool) -> None:
    result = agent.run(task)
    print(result)
    if show_trace:
        print(json.dumps(agent.get_trace(), ensure_ascii=False, indent=2))
