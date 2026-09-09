from __future__ import annotations

from src.analysis.angle_utils import calculate_angle, get_landmark_xy


def _get_knee_angle(landmarks, side):
    hip = get_landmark_xy(landmarks, f"{side}_hip")
    knee = get_landmark_xy(landmarks, f"{side}_knee")
    ankle = get_landmark_xy(landmarks, f"{side}_ankle")
    return calculate_angle(hip, knee, ankle)


def analyze_gait(landmarks):
    """Generate a simple gait summary for left/right limb pattern monitoring."""
    try:
        left_angle = _get_knee_angle(landmarks, "left")
        right_angle = _get_knee_angle(landmarks, "right")
    except (KeyError, TypeError, ValueError):
        return {
            "status": "invalid_pose",
            "score": 0,
            "warnings": ["A full gait pose could not be detected."],
            "suggestions": ["Ensure both lower limbs are visible in the frame."],
            "metrics": {},
        }

    symmetry_gap = abs(left_angle - right_angle)
    score = max(0.0, 100.0 - symmetry_gap * 2.0)

    if symmetry_gap > 12:
        status = "asymmetric_gait"
        warnings = ["The gait pattern shows notable asymmetry between limbs."]
        suggestions = [
            "Encourage equal weight transfer and increase left-right balance during stance.",
            "Watch for a reduced knee bend on the weaker side.",
        ]
    elif score >= 80:
        status = "stable_gait"
        warnings = []
        suggestions = [
            "Maintain a smooth step pattern and keep the trunk aligned over the pelvis.",
        ]
    else:
        status = "needs_attention"
        warnings = ["The gait pattern is less stable than expected."]
        suggestions = [
            "Reduce stride length and focus on controlled knee flexion during stance.",
        ]

    return {
        "status": status,
        "score": round(score, 2),
        "warnings": warnings,
        "suggestions": suggestions,
        "metrics": {
            "left_knee_angle": round(left_angle, 2),
            "right_knee_angle": round(right_angle, 2),
            "symmetry_gap": round(symmetry_gap, 2),
        },
    }
