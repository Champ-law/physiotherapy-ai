from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SessionManager:
    results: List[Dict[str, Any]] = field(default_factory=list)

    def add_result(self, result):
        self.results.append(result)

    @property
    def average_score(self):
        if not self.results:
            return 0.0
        scores = [float(item.get("score", 0)) for item in self.results if "score" in item]
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 2)

    @property
    def summary(self):
        return {
            "count": len(self.results),
            "average_score": self.average_score,
            "results": self.results,
        }
