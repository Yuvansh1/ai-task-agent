"""
Unit tests for AgentController routing and execution logic.
MLflow tracking is disabled during tests via environment variable.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))

# Point MLflow at a temp dir so tests don't write to mlruns
os.environ["MLFLOW_TRACKING_URI"] = "/tmp/mlflow-test-runs"

import pytest
from unittest.mock import patch, MagicMock
from agent import AgentController, AgentResult


class TestAgentRouting:
    def setup_method(self):
        self.agent = AgentController()

    def test_single_step_detection(self):
        assert not self.agent._is_multistep("uppercase hello world")

    def test_multistep_detection_and_then(self):
        assert self.agent._is_multistep("get weather in Toronto and then analyze sentiment")

    def test_multistep_detection_analyze(self):
        assert self.agent._is_multistep("weather in Tokyo and analyze the sentiment")

    def test_extract_primary_task(self):
        task = "uppercase hello and then analyze sentiment"
        primary = self.agent._extract_primary_task(task)
        assert "uppercase hello" in primary
        assert "analyze" not in primary

    def test_select_tool_returns_tuple(self):
        tool, confidence = self.agent._select_tool("uppercase this text")
        assert tool is not None
        assert 0.0 <= confidence <= 1.0

    def test_select_tool_text_processor_for_uppercase(self):
        tool, confidence = self.agent._select_tool("uppercase hello")
        assert tool.name == "TextProcessorTool"

    def test_select_tool_weather_for_weather(self):
        tool, confidence = self.agent._select_tool("what is the weather in Toronto")
        assert "Weather" in tool.name

    def test_count_retries_no_retries(self):
        steps = ["Executing tool", "Executed successfully in 5.0 ms"]
        assert self.agent._count_retries(steps) == 0

    def test_count_retries_with_failure(self):
        steps = ["Attempt 1 failed: some error", "Attempt 2 failed: some error"]
        assert self.agent._count_retries(steps) == 2


class TestAgentRun:
    def setup_method(self):
        self.agent = AgentController()

    def test_run_returns_agent_result(self):
        result = self.agent.run("uppercase hello")
        assert isinstance(result, AgentResult)

    def test_run_single_tool_has_output(self):
        result = self.agent.run("uppercase hello world")
        assert result.output
        assert "HELLO WORLD" in result.output

    def test_run_has_steps(self):
        result = self.agent.run("uppercase hello")
        assert isinstance(result.steps, list)
        assert len(result.steps) > 0

    def test_run_has_tool_used(self):
        result = self.agent.run("uppercase hello")
        assert result.tool_used

    def test_run_chained_task(self):
        result = self.agent.run("uppercase hello and then analyze sentiment")
        assert result.output
        assert "->" in result.tool_used or "Sentiment" in result.tool_used

    def test_result_to_dict(self):
        result = self.agent.run("word count hello world")
        d = result.to_dict()
        assert "output" in d
        assert "steps" in d
        assert "tool_used" in d
        assert "error" in d

    def test_run_no_error_on_valid_task(self):
        result = self.agent.run("lowercase HELLO")
        assert result.error is None
