"""Layered configuration with defaults, YAML values, and environment overrides."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 700,
        "timeout_seconds": 30,
    },
    "system": {"max_agent_steps": 20, "log_level": "INFO", "mode": "two_d"},
    "navigation": {"max_steps": 100, "backend": "two_d"},
    "perception": {"confidence_threshold": 0.5, "backend": "two_d"},
    "memory": {"db_path": "data/memory.db", "decay_days": 30},
    "ai2thor": {
        "scene": "FloorPlan1",
        "width": 640,
        "height": 480,
        "grid_size": 0.25,
        "record_path": None,
    },
    "two_d": {"width": 32, "height": 23, "visibility_radius": 2},
}


class Config:
    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        values = deepcopy(DEFAULT_CONFIG)
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as stream:
                file_values = yaml.safe_load(stream) or {}
            if not isinstance(file_values, dict):
                raise ValueError("Configuration root must be a YAML object.")
            self._merge(values, file_values)
        self._apply_environment_overrides(values)
        self._config = values

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self._config
        for segment in key.split("."):
            if not isinstance(value, dict) or segment not in value:
                return default
            value = value[segment]
        return value

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(self._config)

    @staticmethod
    def _merge(destination: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(destination.get(key), dict):
                Config._merge(destination[key], value)
            else:
                destination[key] = value

    @staticmethod
    def _apply_environment_overrides(values: dict[str, Any]) -> None:
        mappings = {
            "HOMENAV_LLM_MODEL": ("llm", "model"),
            "HOMENAV_LLM_TIMEOUT_SECONDS": ("llm", "timeout_seconds"),
            "HOMENAV_SYSTEM_LOG_LEVEL": ("system", "log_level"),
            "HOMENAV_SYSTEM_MAX_AGENT_STEPS": ("system", "max_agent_steps"),
            "HOMENAV_MEMORY_DB_PATH": ("memory", "db_path"),
        }
        for environment_name, path in mappings.items():
            raw_value = os.getenv(environment_name)
            if raw_value is None:
                continue
            existing = values[path[0]][path[1]]
            if isinstance(existing, bool):
                value: Any = raw_value.lower() in {"1", "true", "yes"}
            elif isinstance(existing, int):
                value = int(raw_value)
            elif isinstance(existing, float):
                value = float(raw_value)
            else:
                value = raw_value
            values[path[0]][path[1]] = value


config = Config()
