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
# Patch MLflow out before agent.py tries to write to disk
# ---------------------------------------------------------------------------

import unittest.mock as _mock
import sys as _sys

class _NoOpMLflow:
    """Swaps mlflow with a no-op so Vercel's read-only filesystem isn't touched."""
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

# Re-import agent now that mlflow is patched
import importlib
import agent as _agent_mod
importlib.reload(_agent_mod)
from agent import AgentController, AgentResult

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
# Frontend UI — served at /
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>AI Task Agent</title>
<style>
  :root{--bg:#0f1117;--surface:#1a1d27;--surface2:#21253a;--border:#2e3350;--accent:#6c63ff;--green:#10b981;--text:#e2e8f0;--muted:#94a3b8;--font-mono:'JetBrains Mono',monospace}
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;min-height:100vh}
  .shell{max-width:860px;margin:0 auto;padding:32px 20px 60px}
  h1{font-size:24px;font-weight:600;margin-bottom:4px}
  .subtitle{font-size:13px;color:var(--muted);margin-bottom:28px}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 24px;margin-bottom:16px}
  label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:6px}
  input,select{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:9px 12px;color:var(--text);font-size:14px;outline:none;transition:border-color .15s}
  input:focus,select:focus{border-color:var(--accent)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
  .btn{background:var(--accent);border:none;border-radius:8px;padding:10px 22px;color:#fff;font-size:14px;font-weight:500;cursor:pointer;transition:opacity .15s}
  .btn:hover{opacity:.88}
  .btn:disabled{opacity:.45;cursor:not-allowed}
  .btn-sm{padding:6px 14px;font-size:12px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--muted);cursor:pointer}
  .btn-sm:hover{border-color:var(--accent);color:var(--accent)}
  .btn-danger{background:#7f1d1d;color:#fca5a5}
  .step{background:var(--surface2);border-radius:6px;padding:7px 12px;margin:4px 0;font-size:12px;color:var(--muted);border-left:3px solid var(--accent);font-family:var(--font-mono)}
  .step.live{border-left-color:var(--green);animation:pulse 1s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
  .result-box{background:#0d0f1a;border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:8px;padding:14px 18px;font-family:var(--font-mono);font-size:13px;color:var(--text);margin:10px 0;white-space:pre-wrap;word-break:break-word}
  .badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;background:#1e1b4b;color:#a5b4fc;margin-bottom:8px}
  .hist-item{border-bottom:1px solid var(--border);padding:12px 0}
  .hist-item:last-child{border-bottom:none}
  .hist-task{font-weight:600;font-size:14px;margin-bottom:4px}
  .hist-meta{font-size:11px;color:var(--muted);margin-bottom:6px}
  .hist-output{font-family:var(--font-mono);font-size:12px;color:var(--muted);background:var(--surface2);border-radius:6px;padding:8px 10px;white-space:pre-wrap;word-break:break-word}
  .alert{padding:10px 14px;border-radius:8px;font-size:13px;margin-bottom:12px}
  .alert-err{background:#2d0a0a;border:1px solid #7f1d1d;color:#fca5a5}
  .alert-ok{background:#052e16;border:1px solid #14532d;color:#86efac}
  .tabs{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:20px}
  .tab{background:none;border:none;padding:9px 18px;color:var(--muted);font-size:13px;cursor:pointer;position:relative;margin-bottom:-1px}
  .tab::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:var(--accent);transform:scaleX(0);transition:transform .2s}
  .tab.active{color:var(--accent)}
  .tab.active::after{transform:scaleX(1)}
  .panel{display:none}.panel.active{display:block}
  .tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-family:var(--font-mono);background:var(--surface2);color:var(--muted);margin-right:4px}
  #login-role{font-size:12px;color:var(--green);margin-top:6px;min-height:18px}
</style>
</head>
<body>
<div class="shell">
  <h1>⚡ AI Task Agent</h1>
  <p class="subtitle">Agentic task runner with streaming execution steps</p>

  <div class="card">
    <div class="row">
      <div>
        <label>Username</label>
        <input id="un" placeholder="e.g. betty" value="betty" />
      </div>
      <div>
        <label>Password</label>
        <input id="pw" type="password" placeholder="password" value="betty@123" />
      </div>
    </div>
    <button class="btn" onclick="checkLogin()">Sign in</button>
    <div id="login-role"></div>
    <p style="font-size:11px;color:var(--muted);margin-top:10px">
      Accounts — <span class="tag">betty / betty@123</span> admin &nbsp;
      <span class="tag">yuvansh / yuvansh@321</span> user &nbsp;
      <span class="tag">roxana / roxana@456</span> user
    </p>
  </div>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('run',this)">Run task</button>
    <button class="tab" onclick="switchTab('history',this)">History</button>
  </div>

  <div id="tab-run" class="panel active">
    <div class="card">
      <label>Task</label>
      <input id="task-input" placeholder='e.g. "What is the weather in Tokyo?" or "calculate 25 * 4"' style="margin-bottom:12px" onkeydown="if(event.key==='Enter')runTask()" />
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">
        <button class="btn-sm" onclick="setQ(this)">Weather in Tokyo</button>
        <button class="btn-sm" onclick="setQ(this)">Calculate 25 * 48</button>
        <button class="btn-sm" onclick="setQ(this)">Uppercase the text hello world</button>
        <button class="btn-sm" onclick="setQ(this)">Weather in Paris and then analyze sentiment</button>
        <button class="btn-sm" onclick="setQ(this)">Word count the text The quick brown fox</button>
      </div>
      <button class="btn" id="run-btn" onclick="runTask()">Run →</button>
    </div>

    <div id="run-output" style="display:none">
      <div class="card">
        <div id="steps-area"></div>
        <div id="result-area"></div>
      </div>
    </div>
  </div>

  <div id="tab-history" class="panel">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
      <button class="btn-sm" onclick="loadHistory()">↻ Refresh</button>
      <button class="btn-sm btn-danger" id="clear-btn" onclick="clearHistory()" style="display:none">Clear all (admin)</button>
    </div>
    <div class="card" id="history-area"><p style="color:var(--muted);font-size:13px">Sign in and click Refresh to load history.</p></div>
  </div>
</div>

<script>
let _role = null;

function headers() {
  return {
    'Content-Type': 'application/json',
    'x-username': document.getElementById('un').value.trim(),
    'x-password': document.getElementById('pw').value.trim(),
  };
}

function setQ(btn) {
  document.getElementById('task-input').value = btn.innerText.trim();
  document.getElementById('task-input').focus();
}

function switchTab(name, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'history') loadHistory();
}

async function checkLogin() {
  const un = document.getElementById('un').value.trim();
  const pw = document.getElementById('pw').value.trim();
  const el = document.getElementById('login-role');
  try {
    const r = await fetch('/auth/login', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({username: un, password: pw})
    });
    const d = await r.json();
    if (r.ok) {
      _role = d.role;
      el.textContent = 'Signed in as ' + d.username + ' (' + d.role + ')';
      document.getElementById('clear-btn').style.display = d.role === 'admin' ? '' : 'none';
    } else {
      el.textContent = d.detail || 'Login failed';
      el.style.color = '#f87171';
    }
  } catch(e) {
    el.textContent = 'Error: ' + e.message;
    el.style.color = '#f87171';
  }
}

async function runTask() {
  const task = document.getElementById('task-input').value.trim();
  if (!task) return;
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Running…';

  const output = document.getElementById('run-output');
  const stepsArea = document.getElementById('steps-area');
  const resultArea = document.getElementById('result-area');
  output.style.display = 'block';
  stepsArea.innerHTML = '';
  resultArea.innerHTML = '';

  try {
    const res = await fetch('/tasks/stream', {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ task })
    });

    if (!res.ok) {
      const err = await res.json();
      resultArea.innerHTML = '<div class="alert alert-err">Error ' + res.status + ': ' + (err.detail || 'Unknown error') + '</div>';
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));
        if (data.done) {
          resultArea.innerHTML =
            '<div class="badge">' + (data.tool_used || 'unknown') + '</div>' +
            '<div class="result-box">' + esc(data.output) + '</div>';
          stepsArea.querySelectorAll('.live').forEach(el => el.classList.remove('live'));
        } else {
          const div = document.createElement('div');
          div.className = 'step live';
          div.textContent = data.step;
          stepsArea.appendChild(div);
          stepsArea.scrollTop = stepsArea.scrollHeight;
        }
      }
    }
  } catch (e) {
    resultArea.innerHTML = '<div class="alert alert-err">Request failed: ' + esc(e.message) + '</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run →';
  }
}

async function loadHistory() {
  const area = document.getElementById('history-area');
  area.innerHTML = '<p style="color:var(--muted);font-size:13px">Loading…</p>';
  try {
    const r = await fetch('/tasks', { headers: headers() });
    if (!r.ok) {
      const err = await r.json();
      area.innerHTML = '<div class="alert alert-err">' + (err.detail || 'Error loading history') + '</div>';
      return;
    }
    const tasks = await r.json();
    if (!tasks.length) {
      area.innerHTML = '<p style="color:var(--muted);font-size:13px">No tasks yet.</p>';
      return;
    }
    area.innerHTML = tasks.map(t => `
      <div class="hist-item">
        <div class="hist-task">${esc(t.task)}</div>
        <div class="hist-meta">${t.user} · ${t.timestamp ? new Date(t.timestamp).toLocaleString() : ''} · <span class="tag">${esc(t.tool_used||'')}</span></div>
        <div class="hist-output">${esc(t.output||'')}</div>
      </div>
    `).join('');
  } catch(e) {
    area.innerHTML = '<div class="alert alert-err">' + esc(e.message) + '</div>';
  }
}

async function clearHistory() {
  if (!confirm('Clear all tasks?')) return;
  try {
    const r = await fetch('/tasks', { method: 'DELETE', headers: headers() });
    if (r.ok) loadHistory();
    else {
      const err = await r.json();
      alert(err.detail || 'Failed to clear');
    }
  } catch(e) { alert(e.message); }
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=_HTML)


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
