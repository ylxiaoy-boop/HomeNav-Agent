"""Optional FastAPI interface following the document's REST response contract."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from interfaces.cli import build_agent

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    HTTPException = None
    BaseModel = None
else:
    class TaskRequest(BaseModel):
        task: str


def create_app() -> Any:
    """Create the REST application only when FastAPI is installed."""
    if FastAPI is None or HTTPException is None or BaseModel is None:
        raise RuntimeError("Install fastapi and uvicorn to use the REST interface.")

    app = FastAPI(title="HomeNav-Agent", version="1.0")
    agent = build_agent()
    tasks: dict[str, dict[str, Any]] = {}

    @app.post("/api/task/start")
    def start_task(request: TaskRequest) -> dict[str, Any]:
        task_id = str(uuid4())
        result = agent.run(request.task)
        tasks[task_id] = {"status": "completed", "result": result, "trace": agent.get_trace()}
        return {"code": 0, "message": "success", "data": {"task_id": task_id, **tasks[task_id]}}

    @app.get("/api/task/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        return {"code": 0, "message": "success", "data": {"task_id": task_id, **task}}

    @app.get("/api/task/{task_id}/trace")
    def get_trace(task_id: str) -> dict[str, Any]:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        return {"code": 0, "message": "success", "data": task["trace"]}

    @app.get("/api/tools")
    def get_tools() -> dict[str, Any]:
        return {"code": 0, "message": "success", "data": agent.tools.get_all_descriptions()}

    @app.get("/api/system/status")
    def get_status() -> dict[str, Any]:
        return {"code": 0, "message": "success", "data": {"mode": "simulated", "tools": agent.tools.get_tool_names()}}

    return app


try:
    app = create_app()
except RuntimeError:
    app = None
