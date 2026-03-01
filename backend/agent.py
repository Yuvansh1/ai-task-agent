"""
AgentController — the brain of the system.

Receives a plain-text task, picks the right tool (or chains two tools
for complex requests), runs it, and returns a structured result with
a full step-by-step execution trace.

Capabilities:
  - Single-tool execution
  - Multi-tool chaining (e.g. weather → sentiment)
  - Automatic retry on tool failure
"""
from __future__ import annotations

import time
from typing import Any

from tools import ALL_TOOLS, BaseTool, SentimentTool, WeatherMockTool, TextProcessorTool


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
      5. Return an AgentResult containing output + full trace.
    """

    _MIN_CONFIDENCE = 0.1
    _MAX_RETRIES = 2

    # Phrases that tell us the user expects two tools to run in sequence
    _CHAIN_TRIGGERS = [
        "and then", "and also", "then analyze", "then tell me",
        "sentiment of the weather", "and sentiment", "and analyze",
        "analyze the", "analyse the", "and check the sentiment",
        "what is the sentiment", "how does it feel",
    ]

    # ── Entry point ─────
    def run(self, task: str) -> AgentResult:
        steps: list[str] = []
        steps.append(f'Received task: "{task}"')

        if self._is_multistep(task):
            steps.append("Multi-step task detected — will chain two tools")
            return self._run_chained(task, steps)
        else:
            steps.append("Single-step task — selecting best tool…")
            return self._run_single(task, steps)

    # ── Standard single-tool path ───────
    def _run_single(self, task: str, steps: list[str]) -> AgentResult:
        tool, confidence = self._select_tool(task)

        if tool is None or confidence < self._MIN_CONFIDENCE:
            steps.append("No suitable tool found — using TextProcessorTool as fallback")
            tool = ALL_TOOLS[0]
        else:
            steps.append(f"Selected tool: {tool.name} (confidence {confidence:.0%})")

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
        """
        Chain two tools together. The output of tool 1 becomes
        part of the input for tool 2.

        Supported chains:
          - Weather → Sentiment  ("weather in Tokyo and analyze the sentiment")
          - Text    → Sentiment  ("uppercase hello and analyze the sentiment")
          - Any     → Sentiment  (fallback: run best tool, then sentiment)
        """
        tools_used = []

        # ── Step 1: run the primary tool 
        # Strip the chaining clause so tool-scoring only sees the first instruction.
        primary_task = self._extract_primary_task(task)
        tool1, confidence1 = self._select_tool(primary_task)

        if tool1 is None or confidence1 < self._MIN_CONFIDENCE:
            tool1 = ALL_TOOLS[0]
            steps.append("Step 1 — No primary tool found, using TextProcessorTool as fallback")
        else:
            steps.append(f"Step 1 — Selected primary tool: {tool1.name} (confidence {confidence1:.0%})")

        result1, error1 = self._execute_with_retry(tool1, primary_task, steps, prefix="Step 1")
        steps.append(f"Step 1 — Result: {result1}")
        tools_used.append(tool1.name)

        # ── Step 2: analyse the first result with SentimentTool ─
        tool2 = SentimentTool()
        steps.append(f"Step 2 — Selected secondary tool: {tool2.name} (analysing Step 1 output)")
        sentiment_input = f"sentiment of {result1}"
        result2, error2 = self._execute_with_retry(tool2, sentiment_input, steps, prefix="Step 2")
        steps.append(f"Step 2 — Result: {result2}")
        tools_used.append(tool2.name)

        # ── Merge both outputs into a single response 
        final_output = f"{result1}\n\n🔍 Sentiment Analysis: {result2}"
        combined_error = error1 or error2
        steps.append("Returning chained result to user")

        return AgentResult(
            output=final_output,
            steps=steps,
            tool_used=" → ".join(tools_used),
            error=combined_error,
        )

    # ── Internal helpers 
    def _is_multistep(self, task: str) -> bool:
        lower = task.lower()
        return any(trigger in lower for trigger in self._CHAIN_TRIGGERS)

    def _extract_primary_task(self, task: str) -> str:
        """Strip everything from the chaining keyword onward, leaving just the first instruction."""
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
        """Run a tool, retrying up to _MAX_RETRIES times on exception.
        Returns a (output_string, error_string_or_None) tuple."""
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
