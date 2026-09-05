"""Run trace-backed comparison and ablation experiments in the 2D home map."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interfaces.cli import build_agent
from memory.knowledge_base import HouseholdKnowledge
from models.two_d_environment import TwoDHomeEnvironment
from models.yolo_model import YOLOModel
from tools.base import BaseTool


@dataclass(frozen=True)
class TaskCase:
    case_id: str
    category: str
    prompt: str
    expected_object: str


CASES = (
    TaskCase("cup", "single_object", "find a cup", "杯子"),
    TaskCase("remote", "single_object", "find a remote control", "遥控器"),
    TaskCase("phone", "single_object", "find a phone", "手机"),
    TaskCase("towel", "single_object", "find a towel", "毛巾"),
    TaskCase("toothbrush", "single_object", "find a toothbrush", "牙刷"),
    TaskCase("keys", "single_object", "find keys", "钥匙"),
    TaskCase("thirsty", "implicit_need", "I am thirsty", "杯子"),
    TaskCase("watch_tv", "implicit_need", "I want to watch TV", "遥控器"),
    TaskCase("brush_teeth", "implicit_need", "I need to brush my teeth", "牙刷"),
)


class RunConfig:
    def __init__(self, memory_db: Path) -> None:
        self.values = {
            "llm.model": "mock",
            "llm.temperature": 0,
            "llm.max_tokens": 700,
            "llm.timeout_seconds": 30,
            "system.mode": "two_d",
            "system.max_agent_steps": 12,
            "navigation.max_steps": 200,
            "perception.confidence_threshold": 0.5,
            "memory.db_path": str(memory_db),
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class KnowledgeOnlyTool(BaseTool):
    """Ablation: common sense remains available, but learned memory is disabled."""

    name = "memory"
    description = "Ablation memory tool with household knowledge but no persistent experience."
    parameters_schema = {
        "type": "object",
        "properties": {"action": {"type": "string"}, "query": {"type": "string"}},
        "required": ["action"],
    }

    def __init__(self) -> None:
        self.knowledge = HouseholdKnowledge()

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        action = str(parameters.get("action", ""))
        if action == "recall":
            query = str(parameters.get("query", ""))
            results = self.knowledge.query(query)
            return {"success": True, "data": {"query": query, "results": results, "count": len(results)}}
        if action == "remember":
            return {"success": True, "data": {"skipped": True, "reason": "memory ablation"}}
        return {"success": True, "data": {"skipped": True, "reason": "memory ablation"}}


class BlindPerceptionTool(BaseTool):
    """Ablation: navigation proceeds but target confirmation is unavailable."""

    name = "perception"
    description = "Ablation perception tool that always reports no confirmed target."
    parameters_schema = {
        "type": "object",
        "properties": {"mode": {"type": "string"}, "target": {"type": "string"}},
        "required": ["mode"],
    }

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        target = str(parameters.get("target", "target"))
        return {
            "success": True,
            "data": {
                "found": False,
                "object": target,
                "confidence": None,
                "objects": [],
                "room": "unknown",
                "position": "unconfirmed",
                "mode": parameters.get("mode"),
                "backend": "two_d_grid_ablation",
                "reason": "perception check disabled",
            },
        }


def run_agent_episode(
    case: TaskCase,
    memory_db: Path,
    *,
    system: str = "homenav_agent_2d",
    knowledge_only: bool = False,
    blind_perception: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    agent = build_agent(RunConfig(memory_db))
    agent.llm.use_mock = True
    agent.llm.client = None
    if knowledge_only:
        agent.tools.register(KnowledgeOnlyTool(), replace=True)
    if blind_perception:
        agent.tools.register(BlindPerceptionTool(), replace=True)
    result = agent.run(case.prompt)
    trace = agent.get_trace()
    environment = agent.environment
    assert isinstance(environment, TwoDHomeEnvironment)
    history = trace.get("tool_history", [])
    found = {environment.normalize_target(item.get("class_name", "")) for item in trace.get("found_objects", [])}
    expected = environment.normalize_target(case.expected_object)
    navigation_steps = sum(int(((entry.get("observation", {}).get("data") or {}).get("steps") or 0)) for entry in history if entry.get("tool_name") == "navigation")
    memory_hit = any(
        item.get("source") == "long_term"
        for entry in history
        if entry.get("tool_name") == "memory" and (entry.get("parameters", {}) or {}).get("action") == "recall"
        for item in ((entry.get("observation", {}).get("data") or {}).get("results") or [])
        if isinstance(item, dict)
    )
    row = {
        "experiment": "agent_episode",
        "system": system,
        "case_id": case.case_id,
        "category": case.category,
        "prompt": case.prompt,
        "expected_object": case.expected_object,
        "success": expected in found,
        "tool_calls": len(history),
        "navigation_steps": navigation_steps,
        "memory_hit": memory_hit,
        "final_room": (trace.get("current_position", {}) or {}).get("room", ""),
        "result": result,
    }
    return row, trace


def run_baseline(case: TaskCase, system: str) -> tuple[dict[str, Any], dict[str, Any]]:
    environment = TwoDHomeEnvironment()
    detector = YOLOModel(environment)
    target = case.expected_object
    events: list[dict[str, Any]] = []
    steps = 0
    if system == "semantic_direct":
        navigation = environment.navigate(target, 200)
        events.append({"tool_name": "navigation", "parameters": {"target": target}, "observation": navigation})
        steps += int((navigation.get("data") or {}).get("steps") or 0)
        detection = detector.detect(target=target)
        events.append({"tool_name": "perception", "parameters": {"target": target}, "observation": {"data": detection}})
    elif system == "pipeline_scan":
        detection = {"objects": []}
        for room in environment.room_order:
            navigation = environment.navigate(room, 200)
            events.append({"tool_name": "navigation", "parameters": {"target": room}, "observation": navigation})
            steps += int((navigation.get("data") or {}).get("steps") or 0)
            detection = detector.detect(target=target)
            events.append({"tool_name": "perception", "parameters": {"target": target}, "observation": {"data": detection}})
            if detection["objects"]:
                break
    else:
        raise ValueError(f"Unknown baseline '{system}'.")
    success = bool(detection["objects"])
    row = {
        "experiment": "baseline",
        "system": system,
        "case_id": case.case_id,
        "category": case.category,
        "prompt": case.prompt,
        "expected_object": case.expected_object,
        "success": success,
        "tool_calls": len(events),
        "navigation_steps": steps,
        "memory_hit": False,
        "final_room": environment.current_view()["room"],
        "result": "target confirmed" if success else "target not confirmed",
    }
    return row, {"task_goal": case.prompt, "tool_history": events, "current_position": environment.current_view()}


def aggregate(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[field]) for field in fields), []).append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        item = {field: value for field, value in zip(fields, key)}
        item.update({
            "episodes": len(values),
            "success_rate": round(sum(bool(value["success"]) for value in values) / len(values), 4),
            "mean_tool_calls": round(statistics.mean(float(value["tool_calls"]) for value in values), 3),
            "mean_navigation_steps": round(statistics.mean(float(value["navigation_steps"]) for value in values), 3),
            "memory_hit_rate": round(sum(bool(value["memory_hit"]) for value in values) / len(values), 4),
        })
        output.append(item)
    return output


def font(size: int, bold: bool = False):
    candidate = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
    try:
        return ImageFont.truetype(candidate, size)
    except OSError:
        return ImageFont.load_default()


def draw_charts(path: Path, baseline: list[dict[str, Any]], categories: list[dict[str, Any]], memory: list[dict[str, Any]]) -> None:
    image = Image.new("RGB", (1500, 850), "#F6F9FC")
    draw = ImageDraw.Draw(image)
    title_font, body_font, label_font = font(30, True), font(18), font(15)
    draw.rectangle((0, 0, 1500, 82), fill="#12355B")
    draw.text((48, 23), "HomeNav-Agent 2D Household Map: Measured Experiment Summary", font=title_font, fill="#FFFFFF")
    charts = [
        ("Success rate: baseline comparison", [(item["system"], item["success_rate"]) for item in baseline], "rate"),
        ("Navigation steps: baseline comparison", [(item["system"], item["mean_navigation_steps"]) for item in baseline], "steps"),
        ("Memory reuse: navigation steps", [(item["phase"], item["mean_navigation_steps"]) for item in memory], "steps"),
    ]
    display_labels = {
        "homenav_agent_2d": "HomeNav\nAgent",
        "pipeline_scan": "Pipeline\nScan",
        "semantic_direct": "Semantic\nDirect",
        "first_run": "First\nrun",
        "repeated_run": "Repeated\nrun",
    }
    colors = ("#2E78B7", "#D97706", "#0F766E")
    for chart_index, ((title, entries, kind), left, color) in enumerate(zip(charts, (55, 540, 1025), colors)):
        draw.rounded_rectangle((left, 115, left + 400, 710), radius=12, fill="#FFFFFF", outline="#D7E2EC")
        draw.text((left + 20, 140), title, font=body_font, fill="#12355B")
        maximum = 1.0 if kind == "rate" else max(1.0, max(float(value) for _, value in entries))
        bottom, available = 630, 390
        count = max(1, len(entries))
        bar_width = 68
        gap = max(24, (330 - bar_width * count) // max(1, count - 1))
        for index, (label, value) in enumerate(entries):
            x = left + 54 + index * (bar_width + gap)
            height = int(float(value) / maximum * available)
            draw.rounded_rectangle((x, bottom - height, x + bar_width, bottom), radius=6, fill=color)
            shown = f"{float(value) * 100:.1f}%" if kind == "rate" else f"{float(value):.1f}"
            draw.text((x, bottom - height - 25), shown, font=label_font, fill="#1E293B")
            draw.multiline_text((x, bottom + 16), display_labels.get(str(label), str(label).replace("_", "\n")), font=label_font, fill="#475569", spacing=2)
        draw.text((left + 20, 688), "Source: executed 2D-grid episodes", font=label_font, fill="#64748B")
    draw.text((55, 755), "Measured scope: discrete 2D map, A* paths, deterministic object visibility, and real Agent tool traces.", font=body_font, fill="#475569")
    draw.text((55, 790), "Not measured: AI2THOR, first-person video, camera detection accuracy, continuous control, or physical robot behavior.", font=body_font, fill="#475569")
    image.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["experiment", "system", "case_id", "category", "prompt", "expected_object", "repetition", "phase", "success", "tool_calls", "navigation_steps", "memory_hit", "final_room", "result"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible 2D HomeNav-Agent experiments.")
    parser.add_argument("--output-dir", default="output/experiments/two_d_grid", help="Directory for results.")
    parser.add_argument("--repetitions", type=int, default=3, help="Independent repetitions per case.")
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        raise ValueError("--repetitions must be at least 1.")

    output_dir = Path(arguments.output_dir)
    run_tag = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    raw_dir = output_dir / "raw" / run_tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    # Agent coverage across explicit objects and indirect household needs.
    for repetition in range(1, arguments.repetitions + 1):
        for case in CASES:
            row, trace = run_agent_episode(case, raw_dir / f"coverage_{case.case_id}_{repetition}.db")
            row.update({"experiment": "task_coverage", "repetition": repetition, "phase": ""})
            rows.append(row); raw.append({"row": row, "trace": trace})

    # Equal explicit-object comparison on the same map.
    explicit = [case for case in CASES if case.category == "single_object"]
    for repetition in range(1, arguments.repetitions + 1):
        for case in explicit:
            for system in ("homenav_agent_2d", "pipeline_scan", "semantic_direct"):
                if system == "homenav_agent_2d":
                    row, trace = run_agent_episode(case, raw_dir / f"baseline_agent_{case.case_id}_{repetition}.db")
                else:
                    row, trace = run_baseline(case, system)
                row.update({"experiment": "baseline_comparison", "repetition": repetition, "phase": ""})
                rows.append(row); raw.append({"row": row, "trace": trace})

    # Repeated phone task: its actual location differs from the default common-sense first guess.
    phone = next(case for case in CASES if case.case_id == "phone")
    for repetition in range(1, arguments.repetitions + 1):
        for system, knowledge_only in (("homenav_agent_2d", False), ("agent_no_persistent_memory", True)):
            db = raw_dir / f"reuse_{system}_{repetition}.db"
            for phase in ("first_run", "repeated_run"):
                row, trace = run_agent_episode(phone, db, system=system, knowledge_only=knowledge_only)
                row.update({"experiment": "memory_reuse", "repetition": repetition, "phase": phase})
                rows.append(row); raw.append({"row": row, "trace": trace})

    # Component ablations: learned memory and target confirmation are independently removed.
    for repetition in range(1, arguments.repetitions + 1):
        for case in explicit:
            for system, knowledge_only, blind_perception in (
                ("agent_no_persistent_memory", True, False),
                ("agent_no_perception_check", False, True),
            ):
                row, trace = run_agent_episode(
                    case,
                    raw_dir / f"ablation_{system}_{case.case_id}_{repetition}.db",
                    system=system,
                    knowledge_only=knowledge_only,
                    blind_perception=blind_perception,
                )
                row.update({"experiment": "ablation", "repetition": repetition, "phase": ""})
                rows.append(row); raw.append({"row": row, "trace": trace})

    write_csv(output_dir / "episode_metrics.csv", rows)
    (output_dir / "raw_traces.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage = aggregate([row for row in rows if row["experiment"] == "task_coverage"], ("category",))
    baseline = aggregate([row for row in rows if row["experiment"] == "baseline_comparison"], ("system",))
    memory_rows = [row for row in rows if row["experiment"] == "memory_reuse"]
    memory = aggregate([row for row in memory_rows if row["system"] == "homenav_agent_2d"], ("phase",))
    memory_by_system = aggregate(memory_rows, ("system", "phase"))
    ablation = aggregate([row for row in rows if row["experiment"] == "ablation"], ("system",))
    draw_charts(output_dir / "experiment_charts.png", baseline, coverage, memory)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_type": "two_d_discrete_household_environment",
        "scope_statement": "Metrics come from actual Agent and baseline runs on the repository's 2D grid map. They validate discrete task planning, A* paths, memory reuse, and target confirmation only.",
        "repetitions": arguments.repetitions,
        "task_cases": [asdict(case) for case in CASES],
        "episode_count": len(rows),
        "summaries": {
            "task_coverage_by_category": coverage,
            "baseline_comparison": baseline,
            "memory_reuse_phone": memory,
            "memory_reuse_phone_by_system": memory_by_system,
            "ablation": ablation,
        },
        "artifacts": {"episode_metrics": "episode_metrics.csv", "raw_traces": "raw_traces.json", "chart": "experiment_charts.png", "raw_database_batch": str(raw_dir.relative_to(output_dir))},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
