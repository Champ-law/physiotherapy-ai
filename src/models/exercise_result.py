from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExerciseResult:
    exercise: str
    status: str
    score: float
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "exercise": self.exercise,
            "status": self.status,
            "score": self.score,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "metrics": self.metrics,
        }


@dataclass
class SessionSummary:
    exercise_results: List[ExerciseResult]

    @property
    def overall_score(self):
        if not self.exercise_results:
            return 0
        return round(sum(item.score for item in self.exercise_results) / len(self.exercise_results), 2)

    def to_dict(self):
        return {
            "overall_score": self.overall_score,
            "exercise_results": [result.to_dict() for result in self.exercise_results],
        }
