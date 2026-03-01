"""
Streamlit UI for the AI Task Agent.

What it does:
  - Login / logout with role-based access (admin vs user)
  - Streams agent execution steps in real time as each one completes
  - Shows task history (admin sees everyone's; users see their own)
  - Clear History is restricted to admin accounts

Run with Docker:
    docker-compose up --build

Stop with Docker:
    Ctrl + C              (same terminal — stops containers)
    docker-compose down   (another terminal — stops and removes containers)

Run locally:
Terminal 1 — start the backend
    cd backend
    python app.py

Terminal 2 — start the frontend
    streamlit run streamlit_app.py
"""
import json
import time
import requests
import streamlit as st
from datetime import datetime

import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Task Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.result-box {
    background: #1a1d27;
    border: 1px solid #2e3350;
    border-left: 4px solid #6c63ff;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #e2e8f0;
    margin: 8px 0;
    white-space: pre-wrap;
}
.step-item {
    background: #21253a;
    border-radius: 6px;
    padding: 7px 12px;
    margin: 4px 0;
    font-size: 0.82rem;
    color: #94a3b8;
    border-left: 3px solid #6c63ff;
}
.step-item.live {
    border-left-color: #10b981;
    animation: pulse 1s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
.tool-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.history-task { font-weight: 600; color: #e2e8f0; }
.history-output {
    font-family: monospace; color: #94a3b8;
    font-size: 0.82rem; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
.error-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 12px 16px;
    color: #ef4444;
}
.admin-badge {
    background: linear-gradient(135deg,#6c63ff,#a78bfa);
    color: white; border-radius: 12px;
    padding: 2px 10px; font-size: 0.72rem; font-weight: 700;
}
.user-badge {
    background: #21253a; color: #94a3b8;
    border: 1px solid #2e3350; border-radius: 12px;
    padding: 2px 10px; font-size: 0.72rem; font-weight: 700;
}
.login-box {
    max-width: 400px; margin: 80px auto;
    background: #1a1d27; border: 1px solid #2e3350;
    border-radius: 12px; padding: 32px;
}
</style>
""", unsafe_allow_html=True)

# ── UI constants ─────────────────────────────────────────────────────────────────
TOOL_META = {
    "TextProcessorTool":             {"icon": "📝", "color": "#6c63ff"},
    "CalculatorTool":                {"icon": "🧮", "color": "#10b981"},
    "WeatherMockTool":               {"icon": "🌤️", "color": "#f59e0b"},
    "SentimentTool":                 {"icon": "🎭", "color": "#ec4899"},
    "WeatherMockTool → SentimentTool": {"icon": "🌤️→🎭", "color": "#f59e0b"},
    "TextProcessorTool → SentimentTool": {"icon": "📝→🎭", "color": "#6c63ff"},
}

EXAMPLES = [
    "uppercase hello world",
    "word count the quick brown fox",
    "reverse abcdef",
    "calculate (12 + 8) * 3",
    "10 divided by 4",
    "weather in Tokyo",
    "weather in London",
    "sentiment of this is absolutely amazing",
    "sentiment of this terrible experience",
    # Multi-step examples
    "weather in Tokyo and analyze the sentiment",
    "weather in Paris and then analyze the sentiment",
    "uppercase hello world and analyze the sentiment",
]

# ── Initialise session state ──────────────────────────────────────────────────────
for key, default in {
    "logged_in": False,
    "username": "",
    "password": "",
    "role": "",
    "history": [],
    "current_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Utility functions ────────────────────────────────────────────────────────────
def auth_headers() -> dict:
    return {
        "x-username": st.session_state.username,
        "x-password": st.session_state.password,
    }


def format_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return iso


def load_history() -> list:
    try:
        r = requests.get(f"{API_URL}/tasks", headers=auth_headers(), timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def clear_history():
    try:
        requests.delete(f"{API_URL}/tasks", headers=auth_headers(), timeout=5)
        st.session_state.history = []
        st.session_state.current_result = None
    except Exception:
        st.error("Failed to clear history")


def render_result(record: dict):
    tool = record.get("tool_used", "Unknown")
    meta = TOOL_META.get(tool, {"icon": "🔧", "color": "#6c63ff"})

    st.markdown(
        f'<span class="tool-badge" style="background:color-mix(in srgb,{meta["color"]} 15%,transparent);'
        f'color:{meta["color"]};border:1px solid color-mix(in srgb,{meta["color"]} 40%,transparent)">'
        f'{meta["icon"]} {tool}</span>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f'**Task:** *"{record["task"]}"*')
    with col2:
        st.caption(f'🕐 {format_time(record.get("timestamp",""))}')

    st.markdown(f'<div class="result-box">{record["output"]}</div>', unsafe_allow_html=True)

    if record.get("error"):
        st.markdown(f'<div class="error-box">⚠️ {record["error"]}</div>', unsafe_allow_html=True)

    with st.expander(f"🔍 Execution Trace ({len(record.get('steps', []))} steps)"):
        for i, step in enumerate(record.get("steps", []), 1):
            st.markdown(
                f'<div class="step-item"><strong style="color:#6c63ff">Step {i}</strong>'
                f'&nbsp; {step}</div>',
                unsafe_allow_html=True,
            )


# ── Streaming task submission ────────────────────────────────────────────────────
def submit_task_streaming(task: str) -> dict | None:
    """
    POST to /tasks/stream and display each step as it arrives.
    Returns the final result dict when done.
    """
    steps_so_far = []
    steps_placeholder = st.empty()
    final_result = None

    try:
        with requests.post(
            f"{API_URL}/tasks/stream",
            json={"task": task},
            headers=auth_headers(),
            stream=True,
            timeout=30,
        ) as resp:
            if resp.status_code == 401:
                st.error("Authentication failed — please log in again.")
                return None
            if resp.status_code == 403:
                st.error("Permission denied.")
                return None
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])

                if not data.get("done"):
                    steps_so_far.append(data["step"])
                    # Re-render the growing step list live
                    html = "".join(
                        f'<div class="step-item"><strong style="color:#6c63ff">Step {i}</strong>'
                        f'&nbsp; {s}</div>'
                        for i, s in enumerate(steps_so_far, 1)
                    )
                    steps_placeholder.markdown(html, unsafe_allow_html=True)
                else:
                    steps_placeholder.empty()
                    final_result = {
                        "id":        data.get("id", ""),
                        "task":      task,
                        "output":    data["output"],
                        "steps":     data["steps"],
                        "tool_used": data["tool_used"],
                        "timestamp": data.get("timestamp", ""),
                        "user":      st.session_state.username,
                        "error":     None,
                    }

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend. Is the FastAPI server running on port 8000?")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    return final_result


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div style="text-align:center;margin-top:60px;">
        <div style="font-size:2.5rem;">⚡</div>
        <h2 style="margin:8px 0 4px;">AI Task Agent</h2>
        <p style="color:#64748b;font-size:0.9rem;">Please sign in to continue</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        username = st.text_input("Username", placeholder="betty / yuvansh / roxana")
        password = st.text_input("Password", type="password", placeholder="Password")

        if st.button("Sign In", type="primary", use_container_width=True):
            if not username or not password:
                st.warning("Please enter both username and password.")
            else:
                try:
                    resp = requests.post(
                        f"{API_URL}/auth/login",
                        json={"username": username, "password": password},
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.password = password
                        st.session_state.role = data["role"]
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach the backend. Is the FastAPI server running?")

        st.markdown("""
        <div style="margin-top:16px;padding:12px;background:#1a1d27;border-radius:8px;
                    border:1px solid #2e3350;font-size:0.8rem;color:#64748b;">
            <strong style="color:#94a3b8">Demo accounts</strong><br>
            🔴 betty / betty@123 &nbsp;·&nbsp; admin (full access)<br>
            🟢 yuvansh / yuvansh@321 &nbsp;·&nbsp; user (own tasks only)<br>
            🟢 roxana / roxana@456 &nbsp;·&nbsp; user (own tasks only)
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP (authenticated)
# ══════════════════════════════════════════════════════════════════════════════

# ── Top header bar ───────────────────────────────────────────────────────────────
h1, h2 = st.columns([6, 1])
with h1:
    st.markdown("## ⚡ AI Task Agent")
    st.caption("Submit any task — the agent figures out which tool to use and shows you every step in real time.")
with h2:
    badge_cls = "admin-badge" if st.session_state.role == "admin" else "user-badge"
    st.markdown(
        f'<div style="text-align:right;padding-top:8px;">'
        f'<span class="{badge_cls}">{st.session_state.role.upper()}</span><br>'
        f'<small style="color:#64748b">{st.session_state.username}</small></div>',
        unsafe_allow_html=True,
    )
    if st.button("Sign Out", use_container_width=True):
        for key in ["logged_in", "username", "password", "role", "history", "current_result"]:
            st.session_state[key] = False if key == "logged_in" else ([] if key == "history" else (None if key == "current_result" else ""))
        st.rerun()

st.divider()

# ── Page layout ──────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

# ── Left column: task input and result display ───────────────────────────────────
with left:
    st.markdown("### 📝 Submit a Task")

    task_input = st.text_area(
        label="Task",
        placeholder="Try: uppercase hello world · calculate 12 * 4 · weather in Tokyo and analyze the sentiment",
        height=100,
        label_visibility="collapsed",
    )

    ex_col, btn_col = st.columns([3, 1])
    with ex_col:
        example = st.selectbox(
            "Quick examples",
            ["— pick an example —"] + EXAMPLES,
            label_visibility="collapsed",
        )
        if example != "— pick an example —":
            task_input = example

    with btn_col:
        run_clicked = st.button("▶ Run", type="primary", use_container_width=True)

    if run_clicked:
        text = task_input.strip()
        if not text:
            st.warning("Please enter a task first.")
        else:
            st.markdown("**⚡ Running — steps will appear below:**")
            result = submit_task_streaming(text)
            if result:
                st.session_state.current_result = result
                ids = {r["id"] for r in st.session_state.history}
                if result["id"] not in ids:
                    st.session_state.history.insert(0, result)

    st.markdown("### 📊 Result")
    if st.session_state.current_result:
        render_result(st.session_state.current_result)
    else:
        st.info("No result yet — submit a task above.", icon="ℹ️")

# ── Right column: task history ───────────────────────────────────────────────────
with right:
    h_col1, h_col2, h_col3 = st.columns([2, 1, 1])
    with h_col1:
        st.markdown("### 🕐 History")
    with h_col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.history = load_history()
    with h_col3:
        # Clear button only visible to admin
        if st.session_state.role == "admin":
            if st.button("🗑️ Clear", use_container_width=True):
                clear_history()
        else:
            st.markdown(
                '<div style="text-align:center;padding-top:6px;'
                'font-size:0.7rem;color:#64748b;">admin only</div>',
                unsafe_allow_html=True,
            )

    if not st.session_state.history:
        st.info("Nothing here yet — submit a task or hit Refresh.", icon="📭")
    else:
        for record in st.session_state.history:
            tool = record.get("tool_used", "Unknown")
            meta = TOOL_META.get(tool, {"icon": "🔧", "color": "#6c63ff"})
            label = f'{meta["icon"]} {record["task"][:45]}{"…" if len(record["task"]) > 45 else ""}'

            with st.expander(label):
                # Show task owner to admin
                if st.session_state.role == "admin":
                    st.caption(f'👤 {record.get("user", "unknown")}')
                st.markdown(
                    f'<p class="history-task">{record["task"]}</p>'
                    f'<p class="history-output">{record["output"]}</p>',
                    unsafe_allow_html=True,
                )
                if st.button("Inspect", key=f"inspect_{record['id']}"):
                    st.session_state.current_result = record
                    st.rerun()

# ── Sidebar: tools reference and settings ────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Tools")
    base_tools = {k: v for k, v in TOOL_META.items() if "→" not in k}
    for tool, meta in base_tools.items():
        st.markdown(
            f'<div style="padding:8px 12px;background:#1a1d27;border-radius:8px;'
            f'border-left:4px solid {meta["color"]};margin-bottom:8px;">'
            f'<strong>{meta["icon"]} {tool.replace("Tool","")}</strong></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("## 🔗 Chained Task Examples")
    multi = [e for e in EXAMPLES if "and" in e]
    for ex in multi:
        st.code(ex, language=None)

    st.divider()
    st.markdown("## ⚙️ Backend")
    if st.button("Health Check"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=3)
            if r.ok:
                st.success("Backend is online ✅")
            else:
                st.error("Backend returned an error")
        except Exception:
            st.error("Cannot reach backend")

    if st.session_state.role == "admin":
        st.divider()
        st.markdown("## 🔐 Admin")
        st.info(f"Signed in as **{st.session_state.username}** (admin)\n\nAdmin accounts have full visibility across all users and can wipe the task history.")
