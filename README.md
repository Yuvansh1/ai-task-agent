# ⚡ AI Task Agent

Type a task in plain English. The agent figures out what to do and shows you every step as it happens.

Built with FastAPI + Streamlit (only using Python)

## 🎥 Demo Video

[![Watch the demo](https://img.youtube.com/vi/xD1oXjy0tow/maxresdefault.jpg)](https://youtu.be/xD1oXjy0tow?si=r8auv_9ysQFhUcIM)

---

## How it works

```
Streamlit (port 8501)
       │
       │  HTTP requests
       ▼
FastAPI Backend (port 8000)
       │
       ▼
  AgentController
  ├── scores every tool
  └── picks the best match
       │
       ├── TextProcessorTool
       ├── CalculatorTool
       ├── WeatherMockTool
       └── SentimentTool
       │
       ▼
  Storage → tasks.json
```

---

## Project Structure

```
AI_Task_Agent_Assignment/
├── docker-compose.yml
├── Dockerfile.frontend
├── .dockerignore
├── streamlit_app.py
└── backend/
    ├── Dockerfile
    ├── app.py            ← all API endpoints
    ├── agent.py          ← tool selection logic
    ├── tools.py          ← the four tools
    ├── storage.py        ← saves history to JSON
    ├── requirements.txt
    └── tasks.json        ← auto-created on first run
```

---

## Run locally

**Prerequisites:** Python 3.11+

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Terminal 1 — start the backend
cd backend
python app.py

# Terminal 2 — start the frontend
streamlit run streamlit_app.py
```

Open **http://localhost:8501**

**Test accounts:**

| Username | Password    | Role  |
|----------|-------------|-------|
| betty    | betty@123   | Admin |
| yuvansh  | yuvansh@321 | User  |
| roxana   | roxana@456  | User  |

**Run tests:**

```bash
cd backend
pytest tests.py -v
# Expected: 19 passing
```

---

## Run with Docker

**Prerequisites:** Docker Desktop installed and running

```bash
# First time (or after code changes)
docker-compose up --build

# After that
docker-compose up
```

| Service      | URL                        |
|--------------|----------------------------|
| Streamlit UI | http://localhost:8501      |
| API docs     | http://localhost:8000/docs |

**To stop:**
- Same terminal → `Ctrl + C`
- Different terminal → `docker-compose down`

`tasks.json` is mounted as a volume so history is never lost.

---

## What you can ask it

| Task | Example |
|------|---------|
| Math | `calculate (12 + 8) * 3` |
| Text | `uppercase hello world` |
| Weather | `weather in Tokyo` |
| Sentiment | `sentiment of this is amazing` |
| Chained | `weather in Tokyo and analyze the sentiment` |

---

## API endpoints

| Method | Endpoint | Auth | What it does |
|--------|----------|------|--------------|
| POST | `/auth/login` | No | Login, returns role |
| POST | `/tasks` | Yes | Run a task |
| POST | `/tasks/stream` | Yes | Run a task, stream steps live |
| GET | `/tasks` | Yes | Get history |
| GET | `/tasks/{id}` | Yes | Get one task |
| DELETE | `/tasks` | Admin | Clear all history |
| GET | `/health` | No | Check if backend is up |
