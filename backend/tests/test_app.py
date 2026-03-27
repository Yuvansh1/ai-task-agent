"""
Basic tests for the AI Task Agent FastAPI backend.
"""
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# ── Health ─────────────────────────────────────────────────────────────────────
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── Auth ───────────────────────────────────────────────────────────────────────
def test_login_valid_admin():
    response = client.post("/auth/login", json={"username": "betty", "password": "betty@123"})
    assert response.status_code == 200
    assert response.json()["role"] == "admin"


def test_login_valid_user():
    response = client.post("/auth/login", json={"username": "yuvansh", "password": "yuvansh@321"})
    assert response.status_code == 200
    assert response.json()["role"] == "user"


def test_login_invalid_credentials():
    response = client.post("/auth/login", json={"username": "betty", "password": "wrong"})
    assert response.status_code == 401


# ── Tasks auth guard ────────────────────────────────────────────────────────────
def test_get_tasks_without_auth():
    response = client.get("/tasks")
    assert response.status_code == 422  # missing required headers


def test_get_tasks_with_invalid_auth():
    response = client.get("/tasks", headers={"x-username": "betty", "x-password": "wrong"})
    assert response.status_code == 401


def test_get_tasks_with_valid_auth():
    response = client.get("/tasks", headers={"x-username": "yuvansh", "x-password": "yuvansh@321"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ── Submit task ────────────────────────────────────────────────────────────────
def test_submit_empty_task():
    response = client.post(
        "/tasks",
        json={"task": "   "},
        headers={"x-username": "yuvansh", "x-password": "yuvansh@321"},
    )
    assert response.status_code == 400


def test_submit_valid_task():
    response = client.post(
        "/tasks",
        json={"task": "summarize hello world"},
        headers={"x-username": "yuvansh", "x-password": "yuvansh@321"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "output" in data


# ── Admin-only delete ──────────────────────────────────────────────────────────
def test_clear_tasks_as_non_admin():
    response = client.delete("/tasks", headers={"x-username": "yuvansh", "x-password": "yuvansh@321"})
    assert response.status_code == 403


def test_clear_tasks_as_admin():
    response = client.delete("/tasks", headers={"x-username": "betty", "x-password": "betty@123"})
    assert response.status_code == 200
