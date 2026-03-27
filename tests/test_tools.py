"""
Unit tests for individual tools.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))

import pytest
from tools import TextProcessorTool, SentimentTool, WeatherMockTool, CalculatorTool


class TestTextProcessorTool:
    def setup_method(self):
        self.tool = TextProcessorTool()

    def test_uppercase(self):
        result = self.tool.execute("uppercase hello world")
        assert "HELLO WORLD" in result

    def test_lowercase(self):
        result = self.tool.execute("lowercase HELLO WORLD")
        assert "hello world" in result

    def test_word_count(self):
        result = self.tool.execute("word count hello world foo")
        assert "3" in result or "word" in result.lower()

    def test_reverse(self):
        result = self.tool.execute("reverse hello")
        assert "olleh" in result

    def test_can_handle_uppercase(self):
        score = self.tool.can_handle("uppercase this text")
        assert score >= 0.4

    def test_can_handle_low_for_weather(self):
        score = self.tool.can_handle("what is the weather in Toronto")
        assert score < 0.5


class TestSentimentTool:
    def setup_method(self):
        self.tool = SentimentTool()

    def test_positive_sentiment(self):
        result = self.tool.execute("I love this, it is amazing and great")
        assert "positive" in result.lower()

    def test_negative_sentiment(self):
        result = self.tool.execute("this is terrible and awful and bad")
        assert "negative" in result.lower()

    def test_neutral_sentiment(self):
        result = self.tool.execute("the cat sat on the mat")
        assert "neutral" in result.lower()

    def test_can_handle_sentiment_task(self):
        score = self.tool.can_handle("analyze the sentiment of this text")
        assert score > 0.5

    def test_returns_string(self):
        result = self.tool.execute("some text")
        assert isinstance(result, str)


class TestWeatherMockTool:
    def setup_method(self):
        self.tool = WeatherMockTool()

    def test_returns_weather_info(self):
        result = self.tool.execute("weather in Toronto")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_can_handle_weather_task(self):
        score = self.tool.can_handle("what is the weather in Toronto")
        assert score >= 0.5

    def test_can_handle_low_for_math(self):
        score = self.tool.can_handle("calculate 2 + 2")
        assert score < 0.5


class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()

    def test_addition(self):
        result = self.tool.execute("calculate 2 + 3")
        assert "5" in result

    def test_can_handle_math(self):
        score = self.tool.can_handle("calculate 10 times 5")
        assert score > 0.5
