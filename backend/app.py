"""
FastAPI backend for the AI Task Agent.

Features:
  POST   /tasks          - submit a task (requires login)
  - GET  /tasks          — get history (admin sees all, user sees own)
  - GET  /tasks/{id}     — get one task (authenticated)
  - DELETE /tasks        — clear all tasks (admin only)
  - POST /tasks/stream   — submit task with real-time step streaming (authenticated)
  - POST /auth/login     — validate credentials, return role
  - GET  /health         — health check (public)
"""
import asyncio
import json
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import AgentController
from storage import Storage
from tools import ALL_TOOLS, SentimentTool

app = FastAPI(title="AI Task Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = Storage()
agent = AgentController()

# ── User store (hardcoded for demo — would use a DB in production) ─────────────
USER_STORE: dict[str, dict] = {
    "betty":    {"password": "betty@123",    "role": "admin"},
    "yuvansh":  {"password": "yuvansh@321",  "role": "user"},
    "roxana":    {"password": "roxana@456",    "role": "user"},
}


# ── Pydantic models ────────────────────────────────────────────────────────────
class TaskRequest(BaseModel):
    task: str
    user: Optional[str] = None   # overridden by auth header in authenticated endpoints


class LoginRequest(BaseModel):
    username: str
    password: str


# ── Authentication dependency ────────────────────────────────────────────────────
def get_current_user(
    x_username: str = Header(..., description="Your username"),
    x_password: str = Header(..., description="Your password"),
) -> dict:
    """
    Validates x-username / x-password request headers.
    Returns a user dict on success, or raises HTTP 401 on failure.
    """
    user = USER_STORE.get(x_username)
    if not user or user["password"] != x_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": x_username, "role": user["role"]}


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ── Auth route ───────────────────────────────────────────────────────────────────
@app.post("/auth/login")
def login(request: LoginRequest):
    """Check username/password and return the user's role."""
    user = USER_STORE.get(request.username)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"username": request.username, "role": user["role"]}


# ── Task routes ──────────────────────────────────────────────────────────────────
@app.post("/tasks")
def submit_task(
    request: TaskRequest,
    current_user: dict = Depends(get_current_user),
):
    """Accept a task string, run the agent, persist the result."""
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")

    result = agent.run(request.task)
    record = storage.save(request.task, result, current_user["username"])
    return record


@app.post("/tasks/stream")
async def stream_task(
    request: TaskRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Submit a task and receive execution steps as a real-time SSE stream.
    Each event: data: {"step": "...", "done": false}
    Final event: data: {"done": true, "output": "...", "steps": [...], "tool_used": "...", "id": "..."}
    """
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")

    task = request.task.strip()

    async def event_generator():
        steps: list[str] = []

        async def emit(step: str):
            steps.append(step)
            yield f"data: {json.dumps({'step': step, 'done': False})}\n\n"
            await asyncio.sleep(0.25)   # brief pause so Streamlit can render each step before the next arrives

        # ── Walk through agent logic manually so we can emit each step live ──
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
                async for chunk in emit("Step 1 — No primary tool found, using TextProcessorTool as fallback"):
                    yield chunk
            else:
                async for chunk in emit(f"Step 1 — Selected: {tool1.name} (confidence {conf1:.0%})"):
                    yield chunk

            async for chunk in emit(f"Step 1 — Executing {tool1.name}…"):
                yield chunk
            try:
                result1 = tool1.execute(primary_task)
            except Exception as e:
                result1 = f"Error: {e}"
            async for chunk in emit(f"Step 1 — Result: {result1}"):
                yield chunk

            tool2 = SentimentTool()
            async for chunk in emit(f"Step 2 — Selected: {tool2.name} (analysing Step 1 output)"):
                yield chunk
            async for chunk in emit(f"Step 2 — Executing {tool2.name}…"):
                yield chunk
            try:
                result2 = tool2.execute(f"sentiment of {result1}")
            except Exception as e:
                result2 = f"Error: {e}"
            async for chunk in emit(f"Step 2 — Result: {result2}"):
                yield chunk

            final_output = f"{result1}\n\n🔍 Sentiment Analysis: {result2}"
            tool_used = f"{tool1.name} → {tool2.name}"

        else:
            async for chunk in emit("Single-step task — selecting best tool…"):
                yield chunk

            tool, confidence = agent._select_tool(task)
            if tool is None or confidence < agent._MIN_CONFIDENCE:
                tool = ALL_TOOLS[0]
                async for chunk in emit("No suitable tool found — using TextProcessorTool as fallback"):
                    yield chunk
            else:
                async for chunk in emit(f"Selected tool: {tool.name} (confidence {confidence:.0%})"):
                    yield chunk

            async for chunk in emit(f"Executing {tool.name}…"):
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

        # Persist result to storage
        from agent import AgentResult
        result_obj = AgentResult(output=str(final_output), steps=steps, tool_used=tool_used)
        record = storage.save(task, result_obj, current_user["username"])

        # Terminal SSE event — carries the full result payload
        yield f"data: {json.dumps({'done': True, 'output': str(final_output), 'steps': steps, 'tool_used': tool_used, 'id': record['id'], 'timestamp': record['timestamp']})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/tasks")
def get_tasks(current_user: dict = Depends(get_current_user)):
    """Return task history. Admins see everything; users see only their own records."""
    if current_user["role"] == "admin":
        return storage.get_all()
    return storage.get_all(user=current_user["username"])


@app.get("/tasks/{task_id}")
def get_task(task_id: str, current_user: dict = Depends(get_current_user)):
    """Look up a single task record by its UUID."""
    record = storage.get_by_id(task_id)
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")
    # Non-admin users must not access other users' records
    if current_user["role"] != "admin" and record.get("user") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return record


@app.delete("/tasks")
def clear_tasks(current_user: dict = Depends(require_admin)):
    """Delete all stored task records. Restricted to admin role."""
    storage.clear()
    return {"message": "All tasks cleared"}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
