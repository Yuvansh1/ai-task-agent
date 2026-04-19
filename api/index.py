"""
AI Task Agent — Vercel-compatible entry point.

Changes vs original backend/app.py:
- sys.path fixed so agent/tools/storage import correctly from project root
- MLflow replaced with in-memory logging (Vercel filesystem is read-only)
- Storage replaced with InMemoryStorage (no file writes on Vercel)
- Streamlit UI replaced with a built-in HTML frontend served at /
- All heavy imports deferred (lazy) so cold start never crashes
"""

import os
import sys

# Ensure the backend folder is on sys.path so agent/tools/storage resolve
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_ROOT, "backend")
for _p in [_ROOT, _BACKEND]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Patch MLflow BEFORE importing agent — agent.py calls mlflow at import time
# and Vercel's filesystem is read-only so mlflow.db write would crash everything
# ---------------------------------------------------------------------------

import sys as _sys

class _NoOpMLflow:
    """No-op mlflow shim — prevents any disk writes on Vercel."""
    def __getattr__(self, name):
        return self._noop
    @staticmethod
    def _noop(*args, **kwargs):
        class _Ctx:
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _Ctx()
    def start_run(self, *a, **kw):
        class _Ctx:
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _Ctx()

_sys.modules["mlflow"] = _NoOpMLflow()

# Now safe to import agent and tools
from agent import AgentController, AgentResult
from tools import ALL_TOOLS, SentimentTool

# ---------------------------------------------------------------------------
# In-memory storage (replaces file-based tasks.json — Vercel is read-only)
# ---------------------------------------------------------------------------

class InMemoryStorage:
    def __init__(self):
        self._data: list[dict[str, Any]] = []

    def save(self, task: str, result: AgentResult, user: str = "anonymous") -> dict:
        record = {
            "id": str(uuid.uuid4()),
            "task": task,
            "user": user,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result.to_dict(),
        }
        self._data.append(record)
        return record

    def get_all(self, user: Optional[str] = None) -> list[dict]:
        records = list(self._data)
        if user:
            records = [r for r in records if r.get("user") == user]
        return list(reversed(records))

    def get_by_id(self, task_id: str) -> Optional[dict]:
        for record in self._data:
            if record["id"] == task_id:
                return record
        return None

    def clear(self) -> None:
        self._data = []

    def export_json(self) -> str:
        return json.dumps(self._data, indent=2)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Task Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = InMemoryStorage()
agent = AgentController()

USER_STORE: dict[str, dict] = {
    "betty":   {"password": "betty@123",   "role": "admin"},
    "yuvansh": {"password": "yuvansh@321", "role": "user"},
    "roxana":  {"password": "roxana@456",  "role": "user"},
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    task: str
    user: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def get_current_user(
    x_username: str = Header(..., description="Your username"),
    x_password: str = Header(..., description="Your password"),
) -> dict:
    user = USER_STORE.get(x_username)
    if not user or user["password"] != x_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": x_username, "role": user["role"]}


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# ---------------------------------------------------------------------------
# Frontend UI — served at / from a separate file (avoids Python string escaping)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ---------------------------------------------------------------------------
# Routes (identical to original app.py)
# ---------------------------------------------------------------------------

@app.post("/auth/login")
def login(request: LoginRequest):
    user = USER_STORE.get(request.username)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"username": request.username, "role": user["role"]}


@app.post("/tasks")
def submit_task(request: TaskRequest, current_user: dict = Depends(get_current_user)):
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")
    result = agent.run(request.task)
    return storage.save(request.task, result, current_user["username"])


@app.post("/tasks/stream")
async def stream_task(request: TaskRequest, current_user: dict = Depends(get_current_user)):
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")

    task = request.task.strip()

    async def event_generator():
        steps: list[str] = []
        start_time = time.perf_counter()

        async def emit(step: str):
            steps.append(step)
            yield f"data: {json.dumps({'step': step, 'done': False})}\n\n"
            await asyncio.sleep(0.15)

        async for chunk in emit(f'Received task: "{task}"'):
            yield chunk

        needs_chain = agent._is_multistep(task)

        if needs_chain:
            async for chunk in emit("Multi-step task detected — will chain two tools"):
                yield chunk

            primary_task = agent._extract_primary_task(task)
            tool1, conf1 = agent._select_tool(primary_task)
            if tool1 is None or conf1 < agent._MIN_CONFIDENCE:
                tool1 = ALL_TOOLS[0]
                async for chunk in emit("Step 1 — No primary tool found, using TextProcessorTool"):
                    yield chunk
            else:
                async for chunk in emit(f"Step 1 — Selected: {tool1.name} (confidence {conf1:.0%})"):
                    yield chunk

            async for chunk in emit(f"Step 1 — Executing {tool1.name}..."):
                yield chunk
            try:
                result1 = tool1.execute(primary_task)
            except Exception as e:
                result1 = f"Error: {e}"
            async for chunk in emit(f"Step 1 — Result: {result1}"):
                yield chunk

            tool2 = SentimentTool()
            async for chunk in emit(f"Step 2 — Selected: {tool2.name}"):
                yield chunk
            try:
                result2 = tool2.execute(f"sentiment of {result1}")
            except Exception as e:
                result2 = f"Error: {e}"
            async for chunk in emit(f"Step 2 — Result: {result2}"):
                yield chunk

            final_output = f"{result1}\n\nSentiment Analysis: {result2}"
            tool_used = f"{tool1.name} -> {tool2.name}"

        else:
            async for chunk in emit("Single-step task — selecting best tool..."):
                yield chunk

            tool, confidence = agent._select_tool(task)
            if tool is None or confidence < agent._MIN_CONFIDENCE:
                tool = ALL_TOOLS[0]
                async for chunk in emit("No suitable tool found — using TextProcessorTool"):
                    yield chunk
            else:
                async for chunk in emit(f"Selected tool: {tool.name} (confidence {confidence:.0%})"):
                    yield chunk

            async for chunk in emit(f"Executing {tool.name}..."):
                yield chunk
            try:
                final_output = tool.execute(task)
            except Exception as e:
                final_output = f"Error: {e}"
            tool_used = tool.name
            async for chunk in emit(f"Result: {final_output}"):
                yield chunk

        async for chunk in emit("Returning result to user"):
            yield chunk

        result_obj = AgentResult(output=str(final_output), steps=steps, tool_used=tool_used)
        record = storage.save(task, result_obj, current_user["username"])

        yield f"data: {json.dumps({'done': True, 'output': str(final_output), 'steps': steps, 'tool_used': tool_used, 'id': record['id'], 'timestamp': record['timestamp']})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/tasks")
def get_tasks(current_user: dict = Depends(get_current_user)):
    if current_user["role"] == "admin":
        return storage.get_all()
    return storage.get_all(user=current_user["username"])


@app.get("/tasks/{task_id}")
def get_task(task_id: str, current_user: dict = Depends(get_current_user)):
    record = storage.get_by_id(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")
    if current_user["role"] != "admin" and record.get("user") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return record


@app.delete("/tasks")
def clear_tasks(current_user: dict = Depends(require_admin)):
    storage.clear()
    return {"message": "All tasks cleared"}


@app.get("/health")
def health():
    return {"status": "ok"}