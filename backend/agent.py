"""
AgentController — the brain of the system.

Receives a plain-text task, picks the right tool (or chains two tools
for complex requests), runs it, and returns a structured result with
a full step-by-step execution trace.

Capabilities:
  - Single-tool execution
  - Multi-tool chaining (e.g. weather → sentiment)
  - Automatic retry on tool failure
  - MLflow experiment tracking per run
"""
from __future__ import annotations

import time
import os
import tempfile
import json
from typing import Any

import mlflow

from tools import ALL_TOOLS, BaseTool, SentimentTool, WeatherMockTool, TextProcessorTool

# Configure MLflow — falls back to local ./mlruns if env var not set
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("ai-task-agent")


class AgentResult:
    def __init__(
        self,
        output: str,
        steps: list[str],
        tool_used: str,
        error: str | None = None,
    ):
        self.output = output
        self.steps = steps
        self.tool_used = tool_used
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "steps": self.steps,
            "tool_used": self.tool_used,
            "error": self.error,
        }


class AgentController:
    """
    Core agent class.

    Workflow:
      1. Accept a plain-text task from the API.
      2. Check whether the task needs two tools chained together.
      3. Score every registered tool via can_handle() and pick the winner.
      4. Execute the chosen tool(s) with built-in retry.
      5. Log params, metrics, and artifacts to MLflow.
      6. Return an AgentResult containing output + full trace.
    """

    _MIN_CONFIDENCE = 0.1
    _MAX_RETRIES = 2

    _CHAIN_TRIGGERS = [
        "and then", "and also", "then analyze", "then tell me",
        "sentiment of the weather", "and sentiment", "and analyze",
        "analyze the", "analyse the", "and check the sentiment",
        "what is the sentiment", "how does it feel",
    ]

    # ── Entry point
    def run(self, task: str) -> AgentResult:
        with mlflow.start_run():
            mlflow.log_param("task_input", task[:250])
            mlflow.log_param("is_multistep", self._is_multistep(task))

            start_time = time.perf_counter()
            steps: list[str] = []
            steps.append(f'Received task: "{task}"')

            if self._is_multistep(task):
                steps.append("Multi-step task detected — will chain two tools")
                result = self._run_chained(task, steps)
            else:
                steps.append("Single-step task — selecting best tool...")
                result = self._run_single(task, steps)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            mlflow.log_metric("execution_time_ms", round(elapsed_ms, 2))
            mlflow.log_metric("step_count", len(result.steps))
            mlflow.log_metric("success", 0 if result.error else 1)
            mlflow.log_metric("retry_count", self._count_retries(result.steps))
            mlflow.set_tag("tool_used", result.tool_used)
            mlflow.set_tag("task_type", "chained" if self._is_multistep(task) else "single")
            mlflow.set_tag("has_error", str(result.error is not None))

            self._log_trace_artifact(task, result, elapsed_ms)

        return result

    # ── Standard single-tool path
    def _run_single(self, task: str, steps: list[str]) -> AgentResult:
        tool, confidence = self._select_tool(task)

        if tool is None or confidence < self._MIN_CONFIDENCE:
            steps.append("No suitable tool found — using TextProcessorTool as fallback")
            tool = ALL_TOOLS[0]
        else:
            steps.append(f"Selected tool: {tool.name} (confidence {confidence:.0%})")
            mlflow.log_param("tool_selected", tool.name)
            mlflow.log_metric("tool_confidence", round(confidence, 4))

        output, error = self._execute_with_retry(tool, task, steps)
        steps.append(f"Result: {output}")
        steps.append("Returning result to user")

        return AgentResult(
            output=str(output),
            steps=steps,
            tool_used=tool.name,
            error=error,
        )

    # ── Chained two-tool path
    def _run_chained(self, task: str, steps: list[str]) -> AgentResult:
        tools_used = []

        primary_task = self._extract_primary_task(task)
        tool1, confidence1 = self._select_tool(primary_task)

        if tool1 is None or confidence1 < self._MIN_CONFIDENCE:
            tool1 = ALL_TOOLS[0]
            steps.append("Step 1 — No primary tool found, using TextProcessorTool as fallback")
        else:
            steps.append(f"Step 1 — Selected primary tool: {tool1.name} (confidence {confidence1:.0%})")
            mlflow.log_param("tool_step1", tool1.name)
            mlflow.log_metric("tool_step1_confidence", round(confidence1, 4))

        result1, error1 = self._execute_with_retry(tool1, primary_task, steps, prefix="Step 1")
        steps.append(f"Step 1 — Result: {result1}")
        tools_used.append(tool1.name)

        tool2 = SentimentTool()
        steps.append(f"Step 2 — Selected secondary tool: {tool2.name} (analysing Step 1 output)")
        mlflow.log_param("tool_step2", tool2.name)

        sentiment_input = f"sentiment of {result1}"
        result2, error2 = self._execute_with_retry(tool2, sentiment_input, steps, prefix="Step 2")
        steps.append(f"Step 2 — Result: {result2}")
        tools_used.append(tool2.name)

        final_output = f"{result1}\n\nSentiment Analysis: {result2}"
        combined_error = error1 or error2
        steps.append("Returning chained result to user")

        return AgentResult(
            output=final_output,
            steps=steps,
            tool_used=" -> ".join(tools_used),
            error=combined_error,
        )

    # ── MLflow artifact helper
    def _log_trace_artifact(self, task: str, result: AgentResult, elapsed_ms: float) -> None:
        trace = {
            "task": task,
            "tool_used": result.tool_used,
            "execution_time_ms": round(elapsed_ms, 2),
            "success": result.error is None,
            "error": result.error,
            "steps": result.steps,
            "output": result.output,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="trace_", delete=False
        ) as f:
            json.dump(trace, f, indent=2)
            tmp_path = f.name

        mlflow.log_artifact(tmp_path, artifact_path="traces")
        os.unlink(tmp_path)

    # ── Internal helpers
    def _is_multistep(self, task: str) -> bool:
        lower = task.lower()
        return any(trigger in lower for trigger in self._CHAIN_TRIGGERS)

    def _extract_primary_task(self, task: str) -> str:
        lower = task.lower()
        for trigger in sorted(self._CHAIN_TRIGGERS, key=len, reverse=True):
            if trigger in lower:
                idx = lower.index(trigger)
                return task[:idx].strip()
        return task

    def _select_tool(self, task: str) -> tuple[BaseTool | None, float]:
        scores = [(tool, tool.can_handle(task)) for tool in ALL_TOOLS]
        scores.sort(key=lambda x: x[1], reverse=True)
        best_tool, best_score = scores[0] if scores else (None, 0.0)
        return best_tool, best_score

    def _execute_with_retry(
        self,
        tool: BaseTool,
        task: str,
        steps: list[str],
        prefix: str = "",
    ) -> tuple[str, str | None]:
        label = f"{prefix} — " if prefix else ""
        error = None
        output = ""

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                attempt_label = f" (attempt {attempt})" if attempt > 1 else ""
                steps.append(f"{label}Executing {tool.name}{attempt_label}")
                t0 = time.perf_counter()
                output = tool.execute(task)
                elapsed = (time.perf_counter() - t0) * 1000
                steps.append(f"{label}Executed successfully in {elapsed:.1f} ms")
                return str(output), None
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                steps.append(f"{label}Attempt {attempt} failed: {error_msg}")
                if attempt == self._MAX_RETRIES:
                    error = error_msg
                    output = f"Error after {self._MAX_RETRIES} attempts: {error_msg}"

        return output, error

    def _count_retries(self, steps: list[str]) -> int:
        return sum(1 for s in steps if "attempt" in s.lower() and "failed" in s.lower())
