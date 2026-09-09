from __future__ import annotations

from src.rules_engine.squat_rules import analyze_squat


class LiveFeedbackProcessor:
    def __init__(self, exercise="squat"):
        self.exercise = exercise

    def analyze_frame(self, landmarks):
        if self.exercise == "squat":
            result = analyze_squat(landmarks)
        else:
            result = {"status": "not_implemented", "score": 0, "suggestions": ["This exercise is not yet implemented."]}

        return {
            "exercise": self.exercise,
            "status": result.get("status", "unknown"),
            "score": result.get("score", 0),
            "suggestions": result.get("suggestions", ["Continue with smooth movement."]),
            "warnings": result.get("warnings", []),
        }
