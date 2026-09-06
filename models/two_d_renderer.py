"""Raster rendering helpers for the two-dimensional household environment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from models.two_d_environment import Cell, TwoDHomeEnvironment


ROOM_COLORS = {
    "客厅": "#E8F1FB",
    "厨房": "#FFF1DD",
    "卧室": "#F1E9FA",
    "卫生间": "#E4F7F4",
    "书房": "#EEF2E7",
}
ROOM_NAMES = {
    "客厅": "Living room",
    "厨房": "Kitchen",
    "卧室": "Bedroom",
    "卫生间": "Bathroom",
    "书房": "Study",
    "走廊": "Hallway",
}
OBJECT_COLORS = {
    "杯子": "#D97706",
    "遥控器": "#2563EB",
    "手机": "#7C3AED",
    "毛巾": "#0F766E",
    "牙刷": "#0F766E",
    "钥匙": "#B45309",
    "书": "#B91C1C",
}


def _font(size: int, bold: bool = False):
    candidates = (
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_two_d_map(
    environment: TwoDHomeEnvironment,
    *,
    agent_cell: Cell | None = None,
    path: Iterable[Cell] | None = None,
    title: str = "HomeNav-Agent 2D Household Environment",
    subtitle: str = "A* grid path, object placement, and live Agent position",
    target: str | None = None,
    width: int = 1280,
    height: int = 720,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "#F7FAFC")
    draw = ImageDraw.Draw(image)
    title_font = _font(31, True)
    subtitle_font = _font(16)
    room_font = _font(19, True)
    object_font = _font(14, True)
    label_font = _font(15)
    grid_left, grid_top = 70, 115
    cell_size = min(31, int((height - grid_top - 48) / environment.height), int((width - 410) / environment.width))
    grid_width = environment.width * cell_size
    grid_height = environment.height * cell_size

    draw.rectangle((0, 0, width, 78), fill="#12355B")
    draw.text((45, 17), title, font=title_font, fill="#FFFFFF")
    draw.text((47, 53), subtitle, font=subtitle_font, fill="#D8E5F1")
    draw.rounded_rectangle((width - 250, 20, width - 45, 57), radius=8, fill="#E7F5F1")
    draw.text((width - 224, 31), "BACKEND: 2D GRID", font=label_font, fill="#0F766E")

    draw.rounded_rectangle((grid_left - 18, grid_top - 18, grid_left + grid_width + 18, grid_top + grid_height + 18), radius=12, fill="#FFFFFF", outline="#D5E1EB", width=2)
    for room, (left, top, right, bottom) in environment.room_rectangles.items():
        x1, y1 = grid_left + left * cell_size, grid_top + top * cell_size
        x2, y2 = grid_left + (right + 1) * cell_size, grid_top + (bottom + 1) * cell_size
        draw.rectangle((x1, y1, x2, y2), fill=ROOM_COLORS[room], outline="#9BB1C4", width=2)
        draw.text((x1 + 10, y1 + 8), room, font=room_font, fill="#12355B")
        draw.text((x1 + 10, y1 + 31), ROOM_NAMES[room], font=label_font, fill="#64748B")

    for x in range(environment.width + 1):
        draw.line((grid_left + x * cell_size, grid_top, grid_left + x * cell_size, grid_top + grid_height), fill="#E5EDF3", width=1)
    for y in range(environment.height + 1):
        draw.line((grid_left, grid_top + y * cell_size, grid_left + grid_width, grid_top + y * cell_size), fill="#E5EDF3", width=1)
    for obstacle in environment._obstacles:
        x, y = obstacle
        draw.rectangle((grid_left + x * cell_size + 3, grid_top + y * cell_size + 3, grid_left + (x + 1) * cell_size - 3, grid_top + (y + 1) * cell_size - 3), fill="#8897A5")

    route = list(path or [])
    if len(route) > 1:
        points = [(grid_left + x * cell_size + cell_size // 2, grid_top + y * cell_size + cell_size // 2) for x, y in route]
        draw.line(points, fill="#F59E0B", width=max(3, cell_size // 5), joint="curve")

    canonical_target = environment.normalize_target(target) if target else None
    for item in environment.objects:
        x, y = item.cell
        center = (grid_left + x * cell_size + cell_size // 2, grid_top + y * cell_size + cell_size // 2)
        color = OBJECT_COLORS.get(item.name, "#475569")
        radius = max(7, cell_size // 3)
        outline = "#DC2626" if canonical_target and environment._matches(canonical_target, item.name) else "#FFFFFF"
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=color, outline=outline, width=3)
        draw.text((center[0] + radius + 3, center[1] - 8), item.name, font=object_font, fill="#1E293B")

    robot = agent_cell or environment.agent_cell
    robot_center = (grid_left + robot[0] * cell_size + cell_size // 2, grid_top + robot[1] * cell_size + cell_size // 2)
    radius = max(9, cell_size // 2 - 2)
    draw.ellipse((robot_center[0] - radius, robot_center[1] - radius, robot_center[0] + radius, robot_center[1] + radius), fill="#0F766E", outline="#FFFFFF", width=3)
    draw.ellipse((robot_center[0] - 4, robot_center[1] - 2, robot_center[0] - 1, robot_center[1] + 1), fill="#FFFFFF")
    draw.ellipse((robot_center[0] + 1, robot_center[1] - 2, robot_center[0] + 4, robot_center[1] + 1), fill="#FFFFFF")
    draw.line((robot_center[0] - 4, robot_center[1] + 6, robot_center[0] + 4, robot_center[1] + 6), fill="#FFFFFF", width=2)
    draw.text((robot_center[0] + radius + 5, robot_center[1] + 3), "Agent", font=object_font, fill="#0F766E")

    panel_left = grid_left + grid_width + 50
    draw.rounded_rectangle((panel_left, 115, width - 48, 615), radius=12, fill="#FFFFFF", outline="#D5E1EB", width=2)
    draw.text((panel_left + 24, 145), "Environment facts", font=_font(22, True), fill="#12355B")
    facts = [
        ("Map", "32 x 23 discrete grid"),
        ("Rooms", "5 functional rooms + hallway"),
        ("Objects", f"{len(environment.objects)} placed items"),
        ("Planner", "A* shortest path"),
        ("Perception", "visible within 2 grid cells"),
        ("Current room", environment.room_for_cell(robot)),
        ("Current cell", f"({robot[0]}, {robot[1]})"),
    ]
    for index, (label, value) in enumerate(facts):
        y = 205 + index * 53
        draw.text((panel_left + 24, y), label, font=label_font, fill="#64748B")
        draw.text((panel_left + 24, y + 20), value, font=_font(16, True), fill="#1E293B")
    draw.rounded_rectangle((panel_left + 24, 545, width - 72, 590), radius=8, fill="#EDF7F5")
    draw.text((panel_left + 38, 560), "Orange: executed A* path | Green: Agent", font=label_font, fill="#0F766E")
    return image
