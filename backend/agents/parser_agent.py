import re
from typing import List, Dict


class ParserAgent:
    """
    Converts raw OCR / ASR / text input into a clean structured math problem.
    """

    def __init__(self):
        self.math_topics = {
            "algebra": ["solve", "equation", "="],
            "probability": ["probability", "coin", "dice", "chance"],
            "calculus": ["limit", "derivative", "differentiate"],
            "linear_algebra": ["matrix", "determinant", "vector"]
        }

    def parse(self, raw_text: str) -> Dict:
        cleaned_text = self._clean_text(raw_text)
        topic = self._detect_topic(cleaned_text)
        variables = self._extract_variables(cleaned_text, topic)
        constraints = self._extract_constraints(cleaned_text)
        needs_clarification = self._needs_clarification(
            cleaned_text, topic, variables
        )

        return {
            "problem_text": cleaned_text,
            "topic": topic,
            "variables": variables,
            "constraints": constraints,
            "needs_clarification": needs_clarification
        }

    # -------------------------
    # TEXT CLEANING
    # -------------------------
    def _clean_text(self, text: str) -> str:
        text = text.lower()

        replacements = {
            "squared": "^2",
            "square": "^2",
            "cube": "^3",
            "equals zero": "= 0",
            "equals to": "=",
            "equal to": "=",
            "equals": "=",
            "approaches": "->"
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        text = re.sub(r"[^a-z0-9=^+\-*/().<> ]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -------------------------
    # TOPIC DETECTION
    # -------------------------
    def _detect_topic(self, text: str) -> str:
        for topic, keywords in self.math_topics.items():
            if any(k in text for k in keywords):
                return topic
        return "algebra"

    # -------------------------
    # VARIABLE EXTRACTION (FIXED)
    # -------------------------
    def _extract_variables(self, text: str, topic: str) -> List[str]:
        # Probability problems often don't have algebraic variables
        if topic == "probability":
            return []

        tokens = re.findall(r"\b[a-z]\b", text)
        return sorted(set(tokens))

    # -------------------------
    # CONSTRAINT EXTRACTION (FIXED)
    # -------------------------
    def _extract_constraints(self, text: str) -> List[str]:
        return re.findall(r"[a-z]\s*(>=|<=|>|<)\s*\d+", text)

    # -------------------------
    # HITL DECISION (FINAL LOGIC)
    # -------------------------
    def _needs_clarification(
        self, text: str, topic: str, variables: List[str]
    ) -> bool:
        # Probability questions are usually self-contained
        if topic == "probability":
            return False

        # Calculus problems like limits/derivatives don't need '='
        if topic == "calculus":
            return False

        # Algebra must have at least one variable and an equation
        if topic == "algebra":
            if not variables:
                return True
            if "=" not in text:
                return True
            return False

        return False
