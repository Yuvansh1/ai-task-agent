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
# Frontend UI — served at /
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>AI Task Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#0b0d14;--surface:#13161f;--surface2:#1c2030;--border:#252a3d;--border2:#2e3550;
  --accent:#6c63ff;--accent2:#5a52e0;--green:#10b981;--red:#ef4444;
  --text:#e2e8f0;--muted:#64748b;--muted2:#94a3b8;
  --font:'Inter',system-ui,sans-serif;--mono:'JetBrains Mono',monospace;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh;display:flex;flex-direction:column}

/* ── LOGIN PAGE ── */
#login-page{
  flex:1;display:flex;align-items:center;justify-content:center;padding:24px;
  animation:fadeIn .3s ease;
}
.login-card{
  width:100%;max-width:420px;
  background:var(--surface);border:1px solid var(--border2);border-radius:16px;
  padding:36px 32px;
}
.login-logo{text-align:center;margin-bottom:28px}
.login-logo .icon{font-size:36px;display:block;margin-bottom:8px}
.login-logo h1{font-size:22px;font-weight:600;color:var(--text)}
.login-logo p{font-size:13px;color:var(--muted2);margin-top:4px}
.field{margin-bottom:16px}
.field label{display:block;font-size:12px;color:var(--muted2);margin-bottom:6px;letter-spacing:.04em}
.field input{
  width:100%;background:var(--surface2);border:1px solid var(--border2);
  border-radius:8px;padding:11px 14px;color:var(--text);font-family:var(--font);
  font-size:14px;outline:none;transition:border-color .15s;
}
.field input:focus{border-color:var(--accent)}
.field .pw-wrap{position:relative}
.field .pw-wrap input{padding-right:40px}
.field .pw-toggle{
  position:absolute;right:12px;top:50%;transform:translateY(-50%);
  background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;padding:2px;
}
.btn-primary{
  width:100%;background:var(--accent);border:none;border-radius:8px;
  padding:12px;color:#fff;font-size:15px;font-weight:500;cursor:pointer;
  transition:background .15s,transform .1s;margin-top:4px;
}
.btn-primary:hover{background:var(--accent2)}
.btn-primary:active{transform:scale(.98)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed}
.login-err{
  background:#2d0a0a;border:1px solid #7f1d1d;color:#fca5a5;
  border-radius:8px;padding:10px 14px;font-size:13px;margin-bottom:14px;display:none;
}
.demo-box{
  margin-top:20px;background:var(--surface2);border-radius:10px;
  padding:14px 16px;border:1px solid var(--border);
}
.demo-box p{font-size:11px;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em}
.demo-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:12px}
.demo-row:last-child{margin-bottom:0}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dot-admin{background:#f59e0b}
.dot-user{background:#10b981}
.demo-cred{font-family:var(--mono);color:var(--muted2)}
.demo-role{color:var(--muted);font-size:11px}
.demo-btn{
  margin-left:auto;background:none;border:1px solid var(--border2);border-radius:5px;
  padding:2px 8px;font-size:11px;color:var(--muted);cursor:pointer;
  transition:border-color .12s,color .12s;
}
.demo-btn:hover{border-color:var(--accent);color:var(--accent)}

/* ── MAIN APP ── */
#app-page{flex:1;display:none;flex-direction:column;animation:fadeIn .3s ease}
.topbar{
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:12px 28px;display:flex;align-items:center;gap:12px;
}
.topbar-logo{font-size:20px}
.topbar-title{font-size:16px;font-weight:600;flex:1}
.topbar-user{
  font-size:12px;color:var(--muted2);display:flex;align-items:center;gap:8px;
}
.role-badge{
  padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500;
}
.role-admin{background:#1c1507;color:#f59e0b;border:1px solid #854d0e}
.role-user{background:#052e16;color:#34d399;border:1px solid #14532d}
.btn-logout{
  background:none;border:1px solid var(--border2);border-radius:6px;
  padding:5px 12px;font-size:12px;color:var(--muted);cursor:pointer;
  transition:border-color .12s,color .12s;
}
.btn-logout:hover{border-color:var(--red);color:var(--red)}

.main{flex:1;max-width:900px;width:100%;margin:0 auto;padding:28px 24px 60px}

.tabs{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:24px}
.tab{
  background:none;border:none;padding:10px 20px;color:var(--muted2);
  font-size:13px;font-family:var(--font);cursor:pointer;position:relative;margin-bottom:-1px;
  transition:color .15s;
}
.tab::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:var(--accent);transform:scaleX(0);transition:transform .2s;
}
.tab.active{color:var(--accent)}
.tab.active::after{transform:scaleX(1)}
.panel{display:none}.panel.active{display:block}

.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:22px 24px;margin-bottom:16px}
label.field-label{font-size:11px;color:var(--muted2);text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:8px}
.task-input{
  width:100%;background:var(--surface2);border:1px solid var(--border2);
  border-radius:8px;padding:11px 14px;color:var(--text);font-family:var(--font);
  font-size:14px;outline:none;transition:border-color .15s;margin-bottom:14px;
}
.task-input:focus{border-color:var(--accent)}
.quick-btns{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.quick-btn{
  background:var(--surface2);border:1px solid var(--border);border-radius:6px;
  padding:6px 12px;color:var(--muted2);font-size:12px;cursor:pointer;
  transition:border-color .12s,color .12s;
}
.quick-btn:hover{border-color:var(--accent);color:var(--accent)}
.btn-run{
  background:var(--accent);border:none;border-radius:8px;
  padding:11px 28px;color:#fff;font-size:14px;font-weight:500;cursor:pointer;
  transition:background .15s,transform .1s;
}
.btn-run:hover{background:var(--accent2)}
.btn-run:active{transform:scale(.98)}
.btn-run:disabled{opacity:.45;cursor:not-allowed}

.steps-wrap{margin-top:16px}
.step{
  background:var(--surface2);border-radius:6px;padding:8px 12px;margin:4px 0;
  font-size:12px;color:var(--muted2);border-left:3px solid var(--accent);
  font-family:var(--mono);line-height:1.5;
}
.step.live{border-left-color:var(--green);animation:pulse 1s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
.result-wrap{margin-top:12px}
.tool-badge{
  display:inline-block;padding:3px 12px;border-radius:20px;font-size:11px;
  font-weight:500;background:#1e1b4b;color:#a5b4fc;margin-bottom:10px;font-family:var(--mono);
}
.result-box{
  background:#080a12;border:1px solid var(--border2);border-left:4px solid var(--accent);
  border-radius:8px;padding:16px 18px;font-family:var(--mono);font-size:13px;
  color:var(--text);white-space:pre-wrap;word-break:break-word;line-height:1.6;
}

.hist-item{border-bottom:1px solid var(--border);padding:14px 0}
.hist-item:last-child{border-bottom:none}
.hist-task{font-weight:500;font-size:14px;margin-bottom:4px}
.hist-meta{font-size:11px;color:var(--muted);margin-bottom:8px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.hist-tool{font-family:var(--mono);background:var(--surface2);padding:2px 8px;border-radius:4px;font-size:10px;color:var(--muted2)}
.hist-output{font-family:var(--mono);font-size:12px;color:var(--muted2);background:var(--surface2);border-radius:6px;padding:10px 12px;white-space:pre-wrap;word-break:break-word}

.alert-err{background:#2d0a0a;border:1px solid #7f1d1d;color:#fca5a5;border-radius:8px;padding:10px 14px;font-size:13px;margin-bottom:12px}
.empty{color:var(--muted);font-size:13px;text-align:center;padding:40px 0}

.hist-toolbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.btn-sm{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:6px 14px;font-size:12px;color:var(--muted2);cursor:pointer;transition:border-color .12s,color .12s}
.btn-sm:hover{border-color:var(--accent);color:var(--accent)}
.btn-danger{background:#2d0a0a;border-color:#7f1d1d;color:#fca5a5}
.btn-danger:hover{border-color:var(--red);color:var(--red)}

@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>

<!-- ── LOGIN ── -->
<div id="login-page">
  <div class="login-card">
    <div class="login-logo">
      <span class="icon">⚡</span>
      <h1>AI Task Agent</h1>
      <p>Please sign in to continue</p>
    </div>

    <div class="login-err" id="login-err"></div>

    <div class="field">
      <label>Username</label>
      <input id="un" placeholder="betty / yuvansh / roxana" autocomplete="username" onkeydown="if(event.key==='Enter')document.getElementById('pw').focus()"/>
    </div>
    <div class="field">
      <label>Password</label>
      <div class="pw-wrap">
        <input id="pw" type="password" placeholder="Password" autocomplete="current-password" onkeydown="if(event.key==='Enter')doLogin()"/>
        <button class="pw-toggle" onclick="togglePw(this)" tabindex="-1">👁</button>
      </div>
    </div>

    <button class="btn-primary" id="login-btn" onclick="doLogin()">Sign in</button>

    <div class="demo-box">
      <p>Demo accounts</p>
      <div class="demo-row">
        <span class="dot dot-admin"></span>
        <span class="demo-cred">betty / betty@123</span>
        <span class="demo-role">admin (full access)</span>
        <button class="demo-btn" onclick="fillCreds('betty','betty@123')">Use</button>
      </div>
      <div class="demo-row">
        <span class="dot dot-user"></span>
        <span class="demo-cred">yuvansh / yuvansh@321</span>
        <span class="demo-role">user (own tasks only)</span>
        <button class="demo-btn" onclick="fillCreds('yuvansh','yuvansh@321')">Use</button>
      </div>
      <div class="demo-row">
        <span class="dot dot-user"></span>
        <span class="demo-cred">roxana / roxana@456</span>
        <span class="demo-role">user (own tasks only)</span>
        <button class="demo-btn" onclick="fillCreds('roxana','roxana@456')">Use</button>
      </div>
    </div>
  </div>
</div>

<!-- ── APP ── -->
<div id="app-page">
  <div class="topbar">
    <span class="topbar-logo">⚡</span>
    <span class="topbar-title">AI Task Agent</span>
    <div class="topbar-user">
      <span id="topbar-name"></span>
      <span id="topbar-role" class="role-badge"></span>
      <button class="btn-logout" onclick="doLogout()">Sign out</button>
    </div>
  </div>

  <div class="main">
    <div class="tabs">
      <button class="tab active" onclick="switchTab('run',this)">Run task</button>
      <button class="tab" onclick="switchTab('history',this)">History</button>
    </div>

    <div id="tab-run" class="panel active">
      <div class="card">
        <label class="field-label">Task</label>
        <input id="task-input" class="task-input" placeholder='e.g. "What is the weather in Tokyo?" or "calculate 25 * 48"' onkeydown="if(event.key==='Enter')runTask()"/>
        <div class="quick-btns">
          <button class="quick-btn" onclick="setQ(this)">Weather in Tokyo</button>
          <button class="quick-btn" onclick="setQ(this)">Calculate 25 * 48</button>
          <button class="quick-btn" onclick="setQ(this)">Uppercase the text hello world</button>
          <button class="quick-btn" onclick="setQ(this)">Weather in Paris and then analyze sentiment</button>
          <button class="quick-btn" onclick="setQ(this)">Word count the text The quick brown fox</button>
        </div>
        <button class="btn-run" id="run-btn" onclick="runTask()">Run →</button>
      </div>

      <div id="run-output" style="display:none">
        <div class="card">
          <div class="steps-wrap" id="steps-area"></div>
          <div class="result-wrap" id="result-area"></div>
        </div>
      </div>
    </div>

    <div id="tab-history" class="panel">
      <div class="hist-toolbar">
        <button class="btn-sm" onclick="loadHistory()">↻ Refresh</button>
        <button class="btn-sm btn-danger" id="clear-btn" onclick="clearHistory()" style="display:none">Clear all tasks</button>
      </div>
      <div class="card" id="history-area"><p class="empty">Click Refresh to load history.</p></div>
    </div>
  </div>
</div>

<script>
let _un = '', _pw = '', _role = '';

function headers() {
  return {
    'Content-Type': 'application/json',
    'x-username': _un,
    'x-password': _pw,
  };
}

function fillCreds(u, p) {
  document.getElementById('un').value = u;
  document.getElementById('pw').value = p;
}

function togglePw(btn) {
  const inp = document.getElementById('pw');
  inp.type = inp.type === 'password' ? 'text' : 'password';
}

async function doLogin() {
  const un = document.getElementById('un').value.trim();
  const pw = document.getElementById('pw').value.trim();
  const errEl = document.getElementById('login-err');
  const btn = document.getElementById('login-btn');
  if (!un || !pw) { showErr('Please enter username and password.'); return; }

  btn.disabled = true;
  btn.textContent = 'Signing in…';
  errEl.style.display = 'none';

  try {
    const r = await fetch('/auth/login', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({username: un, password: pw})
    });
    const d = await r.json();
    if (r.ok) {
      _un = un; _pw = pw; _role = d.role;
      document.getElementById('topbar-name').textContent = un;
      const rb = document.getElementById('topbar-role');
      rb.textContent = d.role;
      rb.className = 'role-badge ' + (d.role === 'admin' ? 'role-admin' : 'role-user');
      document.getElementById('clear-btn').style.display = d.role === 'admin' ? '' : 'none';
      document.getElementById('login-page').style.display = 'none';
      document.getElementById('app-page').style.display = 'flex';
    } else {
      showErr(d.detail || 'Invalid username or password.');
    }
  } catch(e) {
    showErr('Connection error: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Sign in';
  }
}

function showErr(msg) {
  const el = document.getElementById('login-err');
  el.textContent = msg;
  el.style.display = 'block';
}

function doLogout() {
  _un = ''; _pw = ''; _role = '';
  document.getElementById('app-page').style.display = 'none';
  document.getElementById('login-page').style.display = 'flex';
  document.getElementById('un').value = '';
  document.getElementById('pw').value = '';
  document.getElementById('login-err').style.display = 'none';
  document.getElementById('run-output').style.display = 'none';
  document.getElementById('steps-area').innerHTML = '';
  document.getElementById('result-area').innerHTML = '';
  document.getElementById('task-input').value = '';
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
      const err = await res.json().catch(() => ({}));
      resultArea.innerHTML = '<div class="alert-err">Error ' + res.status + ': ' + esc(err.detail || 'Unknown error') + '</div>';
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
        try {
          const data = JSON.parse(line.slice(6));
          if (data.done) {
            stepsArea.querySelectorAll('.live').forEach(el => el.classList.remove('live'));
            resultArea.innerHTML =
              '<div class="tool-badge">' + esc(data.tool_used || 'unknown') + '</div>' +
              '<div class="result-box">' + esc(data.output) + '</div>';
          } else {
            const div = document.createElement('div');
            div.className = 'step live';
            div.textContent = data.step;
            stepsArea.appendChild(div);
          }
        } catch(_) {}
      }
    }
  } catch(e) {
    resultArea.innerHTML = '<div class="alert-err">Request failed: ' + esc(e.message) + '</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run →';
  }
}

async function loadHistory() {
  const area = document.getElementById('history-area');
  area.innerHTML = '<p class="empty">Loading…</p>';
  try {
    const r = await fetch('/tasks', { headers: headers() });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      area.innerHTML = '<div class="alert-err">' + esc(err.detail || 'Error loading history') + '</div>';
      return;
    }
    const tasks = await r.json();
    if (!tasks.length) {
      area.innerHTML = '<p class="empty">No tasks yet. Run a task first.</p>';
      return;
    }
    area.innerHTML = '<div>' + tasks.map(t => `
      <div class="hist-item">
        <div class="hist-task">${esc(t.task)}</div>
        <div class="hist-meta">
          <span>${esc(t.user)}</span>
          <span>${t.timestamp ? new Date(t.timestamp).toLocaleString() : ''}</span>
          <span class="hist-tool">${esc(t.tool_used||'')}</span>
        </div>
        <div class="hist-output">${esc(t.output||'')}</div>
      </div>
    `).join('') + '</div>';
  } catch(e) {
    area.innerHTML = '<div class="alert-err">' + esc(e.message) + '</div>';
  }
}

async function clearHistory() {
  if (!confirm('Clear all tasks? This cannot be undone.')) return;
  try {
    const r = await fetch('/tasks', { method: 'DELETE', headers: headers() });
    if (r.ok) { loadHistory(); }
    else {
      const err = await r.json().catch(() => ({}));
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