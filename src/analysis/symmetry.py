from __future__ import annotations


def analyze_symmetry(metrics):
    left_value = float(metrics.get("left_knee_angle", 0.0))
    right_value = float(metrics.get("right_knee_angle", 0.0))

    diff = abs(left_value - right_value)
    score = max(0.0, 100.0 - diff * 2.0)

    warnings = []
    if diff > 10:
        warnings.append("Marked asymmetry between left and right limbs.")
    elif diff > 5:
        warnings.append("Moderate asymmetry detected.")

    suggestions = []
    if not warnings:
        suggestions.append("Symmetry is within the expected range.")
    else:
        suggestions.append("Balance the range of motion between the left and right sides.")

    return {
        "symmetry_score": round(score, 2),
        "left_value": left_value,
        "right_value": right_value,
        "difference": round(diff, 2),
        "warnings": warnings,
        "suggestions": suggestions,
    }
