"""Run one reproducible AI2-THOR household-navigation validation episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interfaces.cli import build_agent
from models.ai2thor_environment import AI2THORUnavailableError


class ValidationConfig:
    """Minimal immutable configuration for a deterministic validation run."""

    def __init__(self, output_dir: Path, scene: str, width: int, height: int) -> None:
        self.values: dict[str, Any] = {
            "llm.model": "mock",
            "llm.temperature": 0,
            "llm.max_tokens": 700,
            "llm.timeout_seconds": 30,
            "system.mode": "ai2thor",
            "system.max_agent_steps": 12,
            "system.log_level": "INFO",
            "navigation.max_steps": 100,
            "perception.confidence_threshold": 0.5,
            "memory.db_path": str(output_dir / "validation_memory.db"),
            "ai2thor.scene": scene,
            "ai2thor.width": width,
            "ai2thor.height": height,
            "ai2thor.grid_size": 0.25,
            "ai2thor.record_path": str(output_dir / "validation.mp4"),
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record an AI2-THOR HomeNav-Agent validation episode.")
    parser.add_argument("--task", default="find a cup", help="Natural-language household task.")
    parser.add_argument("--scene", default="FloorPlan1", help="AI2-THOR iTHOR scene name.")
    parser.add_argument("--output-dir", default="output/ai2thor_validation", help="Artifact directory.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    arguments = parser.parse_args()

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ValidationConfig(output_dir, arguments.scene, arguments.width, arguments.height)

    try:
        agent = build_agent(config)
    except AI2THORUnavailableError as error:
        print(f"AI2-THOR validation cannot start: {error}")
        return 2

    agent.llm.use_mock = True
    agent.llm.client = None
    try:
        result = agent.run(arguments.task)
        trace = agent.get_trace()
        summary = {
            "task": arguments.task,
            "scene": arguments.scene,
            "result": result,
            "tool_sequence": [item["tool_name"] for item in trace.get("tool_history", [])],
            "trace": trace,
            "video": "validation.mp4",
            "perception_backend": "ai2thor_metadata",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        agent.environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
