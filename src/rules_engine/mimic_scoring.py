from __future__ import annotations

import math


def score_mimic(actual, reference):
    """Compare user landmarks to a reference pose using average point distance."""
    if not actual or not reference:
        return {"score": 0.0, "similarity": "invalid", "warnings": ["No comparison data available."]}

    total_distance = 0.0
    count = 0

    for key, ref_point in reference.items():
        if key not in actual:
            continue
        act_point = actual[key]
        delta_x = act_point[0] - ref_point[0]
        delta_y = act_point[1] - ref_point[1]
        total_distance += math.hypot(delta_x, delta_y)
        count += 1

    if count == 0:
        return {"score": 0.0, "similarity": "invalid", "warnings": ["No common landmarks were available for comparison."]}

    avg_distance = total_distance / count
    score = max(0.0, 100.0 - (avg_distance * 100.0))

    similarity = "high" if score >= 85 else "moderate" if score >= 60 else "low"
    return {
        "score": round(score, 2),
        "similarity": similarity,
        "average_distance": round(avg_distance, 4),
        "warnings": [] if score >= 60 else ["Movement deviates from the reference pose."],
    }
