"""Navigation model adapter with a deterministic local fallback."""

from __future__ import annotations

from typing import Any

from typing import Protocol

from models.simulated_environment import SimulatedHomeEnvironment


class NavigationEnvironment(Protocol):
    def navigate(self, target: str, max_steps: int) -> dict[str, Any]: ...

    def current_view(self) -> dict[str, Any]: ...


class VLFMModel:
    """Expose a stable semantic-navigation contract independent of a VLFM backend."""

    def __init__(self, environment: NavigationEnvironment | None = None) -> None:
        self.environment = environment or SimulatedHomeEnvironment()

    def navigate_to(self, target: str, max_steps: int = 100) -> dict[str, Any]:
        return self.environment.navigate(target, max_steps)

    def get_position(self) -> dict[str, Any]:
        return {"success": True, "data": self.environment.current_view()}
