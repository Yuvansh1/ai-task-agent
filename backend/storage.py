"""
Simple JSON-file persistence for task history.
Thread-safe for single-process usage.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from agent import AgentResult

STORAGE_FILE = os.environ.get("STORAGE_FILE", "tasks.json")


class Storage:
    def __init__(self, filepath: str = STORAGE_FILE):
        self._filepath = filepath
        self._lock = threading.Lock()
        self._data: list[dict[str, Any]] = self._load()

    
    def save(self, task: str, result: AgentResult, user: str = "anonymous") -> dict:
        record = {
            "id": str(uuid.uuid4()),
            "task": task,
            "user": user,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result.to_dict(),
        }
        with self._lock:
            self._data.append(record)
            self._persist()
        return record

    def get_all(self, user: Optional[str] = None) -> list[dict]:
        with self._lock:
            records = list(self._data)
        if user:
            records = [r for r in records if r.get("user") == user]
        return list(reversed(records))  # newest first

    def get_by_id(self, task_id: str) -> Optional[dict]:
        with self._lock:
            for record in self._data:
                if record["id"] == task_id:
                    return record
        return None

    def clear(self) -> None:
        with self._lock:
            self._data = []
            self._persist()

    def export_json(self) -> str:
        with self._lock:
            return json.dumps(self._data, indent=2)

    
    def _load(self) -> list[dict]:
        if os.path.exists(self._filepath):
            try:
                with open(self._filepath) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _persist(self) -> None:
        try:
            with open(self._filepath, "w") as f:
                json.dump(self._data, f, indent=2)
        except OSError:
            pass  # Non-critical — in-memory data is still valid
