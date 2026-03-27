"""
Tool implementations for the Agent system.
Each tool inherits from BaseTool and implements execute().
"""
import re
import math
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def execute(self, input_text: str) -> Any:
        pass

    def can_handle(self, task: str) -> float:
        """Return a confidence score 0-1 for how well this tool fits the task."""
        return 0.0

# TextProcessorTool
class TextProcessorTool(BaseTool):
    name = "TextProcessorTool"
    description = "Processes text: uppercase, lowercase, word count, reverse, etc."

    _KEYWORDS = [
        "uppercase", "upper case", "caps", "capital",
        "lowercase", "lower case", "small",
        "word count", "count words", "how many words",
        "reverse", "character count", "length",
        "trim", "strip",
    ]

    def can_handle(self, task: str) -> float:
        lower = task.lower()
        hits = sum(1 for kw in self._KEYWORDS if kw in lower)
        return min(hits * 0.4, 1.0)

    def execute(self, input_text: str) -> str:
        lower = input_text.lower()

        if any(k in lower for k in ["uppercase", "upper case", "caps", "capital"]):
            text = self._extract_text(input_text, ["uppercase", "upper", "caps", "to caps"])
            return text.upper()

        if any(k in lower for k in ["lowercase", "lower case", "small letters"]):
            text = self._extract_text(input_text, ["lowercase", "lower", "small"])
            return text.lower()

        if any(k in lower for k in ["word count", "count words", "how many words"]):
            text = self._extract_text(input_text, ["word count", "count words", "count the words in", "how many words"])
            words = text.split()
            return f"Word count: {len(words)} words"

        if "reverse" in lower:
            text = self._extract_text(input_text, ["reverse"])
            return text[::-1]

        if any(k in lower for k in ["character count", "char count", "length"]):
            text = self._extract_text(input_text, ["character count", "char count", "length of"])
            return f"Character count: {len(text)} characters"

        # Default: return text stats summary
        words = input_text.split()
        return f"Text stats — words: {len(words)}, characters: {len(input_text)}"

    def _extract_text(self, raw: str, keywords: list) -> str:
        """Strip leading keywords/connectors to isolate the target text."""
        text = raw
        for kw in sorted(keywords, key=len, reverse=True):
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            text = pattern.sub("", text, count=1)
        fillers = r"^[\s:,\-]*?(the text|the string|this text|this string|the following|of|for|:)?\s*"
        text = re.sub(fillers, "", text, flags=re.IGNORECASE).strip()
        return text if text else raw



# CalculatorTool

class CalculatorTool(BaseTool):
    name = "CalculatorTool"
    description = "Evaluates arithmetic expressions: +, -, *, /, ^, sqrt, etc."

    _KEYWORDS = [
        "calculate", "compute", "math", "solve",
        "+", "-", "*", "/", "^", "sqrt", "square root",
        "plus", "minus", "times", "divided", "multiply",
        "what is", "equals", "result of",
    ]
    _NUMBER_RE = re.compile(r"\d")

    def can_handle(self, task: str) -> float:
        lower = task.lower()
        has_number = bool(self._NUMBER_RE.search(task))
        keyword_hits = sum(1 for kw in self._KEYWORDS if kw in lower)
        score = 0.0
        if has_number:
            score += 0.3
        score += min(keyword_hits * 0.25, 0.7)
        return min(score, 1.0)

    def execute(self, input_text: str) -> str:
        expression = self._clean_expression(input_text)
        try:
            result = self._safe_eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Could not evaluate '{expression}': {e}"

    def _clean_expression(self, text: str) -> str:
        """Attempt to extract a math expression from natural language."""
        t = text.lower()
        replacements = {
            "square root of": "sqrt(",
            "sqrt of": "sqrt(",
            "multiplied by": "*",
            "divided by": "/",
            "to the power of": "**",
            "squared": "**2",
            "plus": "+",
            "minus": "-",
            "times": "*",
        }
        for word, sym in sorted(replacements.items(), key=lambda x: -len(x[0])):
            t = t.replace(word, sym)
        prefix_patterns = [
            r"^(what\s+is|calculate|compute|evaluate|solve|find|the\s+result\s+of|result\s+of)\s+",
        ]
        for pat in prefix_patterns:
            t = re.sub(pat, "", t, flags=re.IGNORECASE)
        expr = re.sub(r"[^0-9+\-*/().sqrt% ]", " ", t)
        expr = re.sub(r"\s+", "", expr).strip()
        expr = expr.replace("^", "**")
        return expr

    def _safe_eval(self, expression: str):
        """Safely evaluate a numeric expression with no access to builtins."""
        allowed = {"sqrt": math.sqrt, "abs": abs, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        if isinstance(result, float):
            return round(result, 6)
        return result



# WeatherMockTool

class WeatherMockTool(BaseTool):
    name = "WeatherMockTool"
    description = "Returns mock weather information for a given city."

    # Season-neutral, climatically reasonable mock temperatures
    _MOCK_DATA = {
        "new york":      {"temp": 14, "condition": "Partly Cloudy", "humidity": 60, "wind": "14 km/h NW"},
        "london":        {"temp": 13, "condition": "Overcast",      "humidity": 78, "wind": "20 km/h SW"},
        "tokyo":         {"temp": 18, "condition": "Sunny",         "humidity": 55, "wind": "8 km/h E"},
        "sydney":        {"temp": 22, "condition": "Clear",         "humidity": 50, "wind": "12 km/h SE"},
        "paris":         {"temp": 14, "condition": "Light Rain",    "humidity": 82, "wind": "16 km/h W"},
        "dubai":         {"temp": 35, "condition": "Sunny",         "humidity": 35, "wind": "10 km/h N"},
        "toronto":       {"temp": 10, "condition": "Cloudy",        "humidity": 65, "wind": "18 km/h NW"},
        "etobicoke":     {"temp": 10, "condition": "Cloudy",        "humidity": 65, "wind": "18 km/h NW"},
        "scarborough":   {"temp": 10, "condition": "Partly Cloudy", "humidity": 63, "wind": "16 km/h NW"},
        "north york":    {"temp": 10, "condition": "Cloudy",        "humidity": 66, "wind": "17 km/h NW"},
        "mississauga":   {"temp":  9, "condition": "Overcast",      "humidity": 67, "wind": "19 km/h W"},
        "berlin":        {"temp": 12, "condition": "Foggy",         "humidity": 88, "wind": "6 km/h S"},
        "singapore":     {"temp": 31, "condition": "Thunderstorm",  "humidity": 90, "wind": "22 km/h NE"},
        "san francisco": {"temp": 16, "condition": "Foggy",         "humidity": 75, "wind": "24 km/h W"},
        "chicago":       {"temp": 11, "condition": "Windy",         "humidity": 58, "wind": "35 km/h N"},
        "mumbai":        {"temp": 33, "condition": "Humid",         "humidity": 92, "wind": "15 km/h SW"},
        "vancouver":     {"temp": 13, "condition": "Rainy",         "humidity": 85, "wind": "22 km/h SW"},
        "montreal":      {"temp":  9, "condition": "Cloudy",        "humidity": 70, "wind": "20 km/h NE"},
        "ottawa":        {"temp":  8, "condition": "Clear",         "humidity": 60, "wind": "15 km/h W"},
        "calgary":       {"temp": 10, "condition": "Sunny",         "humidity": 40, "wind": "25 km/h NW"},
        "los angeles":   {"temp": 24, "condition": "Sunny",         "humidity": 45, "wind": "10 km/h W"},
        "miami":         {"temp": 28, "condition": "Humid",         "humidity": 88, "wind": "18 km/h SE"},
        "beijing":       {"temp": 17, "condition": "Hazy",          "humidity": 55, "wind": "12 km/h N"},
        "seoul":         {"temp": 15, "condition": "Clear",         "humidity": 50, "wind": "14 km/h NW"},
        "amsterdam":     {"temp": 12, "condition": "Overcast",      "humidity": 80, "wind": "28 km/h W"},
        "rome":          {"temp": 20, "condition": "Sunny",         "humidity": 52, "wind": "8 km/h SW"},
    }
    _DEFAULT = {"temp": 20, "condition": "Clear", "humidity": 60, "wind": "10 km/h N"}

    _KEYWORDS = ["weather", "temperature", "forecast", "climate", "rain", "sunny", "degrees", "humid"]

    # FIX: strip province/state/country with OR without a preceding comma
    # e.g. "Toronto, ON" and "Toronto ON" both reduce to "Toronto"
    _SUFFIX_RE = re.compile(
        r"[,\s]+\b(ON|BC|AB|QC|MB|SK|NS|NB|PE|NL|NT|YT|NU"
        r"|CA|Canada|USA|US|UK|Australia|Germany|France|Japan"
        r"|Ontario|British Columbia|Alberta|Quebec"
        r"|New South Wales|England|Scotland)\b.*$",
        re.IGNORECASE,
    )

    def can_handle(self, task: str) -> float:
        lower = task.lower()
        hits = sum(1 for kw in self._KEYWORDS if kw in lower)
        return min(hits * 0.5, 1.0)

    def execute(self, input_text: str) -> str:
        city = self._extract_city(input_text)
        lookup = city.lower()
        data = self._MOCK_DATA.get(lookup, self._DEFAULT)
        known = lookup in self._MOCK_DATA
        note = "*(mock data)*" if known else "*(unknown city — showing default mock data)*"
        return (
            f"Weather for {city.title()}: {data['condition']}, "
            f"{data['temp']}°C, Humidity {data['humidity']}%, "
            f"Wind {data['wind']}  {note}"
        )

    def _extract_city(self, text: str) -> str:
        lower = text.lower()

        for kw in ["weather in", "weather for", "forecast for", "temperature in", "how's the weather in"]:
            if kw in lower:
                idx = lower.index(kw) + len(kw)
                raw_city = text[idx:].strip().rstrip("?.")
                # Strip province/state/country suffix (with or without comma)
                raw_city = self._SUFFIX_RE.sub("", raw_city).strip().rstrip(",").strip()
                return raw_city or "Unknown City"

        # Try to find a known city name anywhere in the text (longest first)
        for city in sorted(self._MOCK_DATA, key=len, reverse=True):
            if city in lower:
                return city.title()

        # Fallback: last word(s) before any comma
        first_segment = text.split(",")[0].strip()
        words = [w.strip("?.,!") for w in first_segment.split() if w.strip("?.,!")]
        return " ".join(words[-2:]) if len(words) >= 2 else (words[-1] if words else "Unknown City")



# SentimentTool

class SentimentTool(BaseTool):
    name = "SentimentTool"
    description = "Estimates the sentiment (positive/negative/neutral) of a text."

    _POSITIVE = {"good", "great", "excellent", "amazing", "love", "wonderful", "fantastic",
                 "happy", "best", "awesome", "brilliant", "superb", "nice", "joy", "beautiful"}
    _NEGATIVE = {"bad", "terrible", "awful", "hate", "horrible", "worst", "sad", "disgusting",
                 "poor", "dreadful", "ugly", "angry", "disappointing", "boring"}
    _KEYWORDS = ["sentiment", "tone", "feeling", "mood", "positive or negative", "analyse the text"]

    def can_handle(self, task: str) -> float:
        lower = task.lower()
        hits = sum(1 for kw in self._KEYWORDS if kw in lower)
        return min(hits * 0.6, 1.0)

    def execute(self, input_text: str) -> str:
        text = self._extract_text(input_text)
        words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
        pos = len(words & self._POSITIVE)
        neg = len(words & self._NEGATIVE)
        if pos > neg:
            label, score = "Positive 😊", pos / max(len(words), 1)
        elif neg > pos:
            label, score = "Negative 😞", neg / max(len(words), 1)
        else:
            label, score = "Neutral 😐", 0.5
        return f"Sentiment: {label} (confidence: {score:.0%})"

    def _extract_text(self, raw: str) -> str:
        for kw in ["sentiment of", "analyze", "analyse", "the tone of", "feeling of"]:
            if kw in raw.lower():
                idx = raw.lower().index(kw) + len(kw)
                return raw[idx:].strip().strip('"\'')
        return raw


# Registry of all available tools
ALL_TOOLS: list = [
    TextProcessorTool(),
    CalculatorTool(),
    WeatherMockTool(),
    SentimentTool(),
]
