# Deploying AI Task Agent to Vercel

## Steps

1. Push this repo to GitHub

2. Go to https://vercel.com/new and import the repo

3. Leave all build settings as default — Vercel auto-detects `vercel.json`

4. Click **Deploy**

5. Visit your deployment URL — the full UI is served at `/`

---

## What changed for Vercel

| Original | Vercel version |
|---|---|
| MLflow writing to `mlflow.db` | MLflow patched to no-op (read-only filesystem) |
| `tasks.json` file storage | In-memory storage (resets on cold start) |
| Streamlit frontend | Built-in HTML UI served at `/` from `api/index.py` |
| Docker + docker-compose | `vercel.json` + `@vercel/python` |
| `backend/requirements.txt` with `mlflow`, `streamlit` | Root `requirements.txt` with only `fastapi`, `uvicorn`, `pydantic` |

## Project structure

```
.
├── api/
│   └── index.py          # Vercel entry point — FastAPI app + HTML UI
├── backend/
│   ├── agent.py          # Patched — MLflow is optional/no-op
│   ├── tools.py          # Unchanged
│   └── storage.py        # Not used on Vercel (InMemoryStorage in api/index.py)
├── requirements.txt      # Vercel reads this from root
├── vercel.json           # Routes all traffic to api/index.py
└── .vercelignore         # Excludes Docker, MLflow data, Streamlit, tests
```

## Credentials

| Username | Password | Role |
|---|---|---|
| betty | betty@123 | admin |
| yuvansh | yuvansh@321 | user |
| roxana | roxana@456 | user |

## Notes

- **In-memory state** resets between cold starts. For persistence add Upstash Redis or Supabase.
- **MLflow tracking** is disabled on Vercel. To re-enable, set `MLFLOW_TRACKING_URI` as an environment variable pointing to a hosted MLflow server.
- **Vercel Hobby** has a 10-second function timeout. The streaming endpoint should complete well within this.
