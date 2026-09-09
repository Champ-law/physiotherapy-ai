from __future__ import annotations

from src.analysis.angle_utils import calculate_angle, get_landmark_xy


def _get_arm_angle(landmarks, side):
    shoulder = get_landmark_xy(landmarks, f"{side}_shoulder")
    elbow = get_landmark_xy(landmarks, f"{side}_elbow")
    wrist = get_landmark_xy(landmarks, f"{side}_wrist")
    return calculate_angle(shoulder, elbow, wrist)


def analyze_upper_limb(landmarks):
    """Assess upper-limb reach and compensation using elbow angle consistency."""
    try:
        left_angle = _get_arm_angle(landmarks, "left")
        right_angle = _get_arm_angle(landmarks, "right")
    except (KeyError, TypeError, ValueError):
        return {
            "status": "invalid_pose",
            "score": 0,
            "warnings": ["The upper limb could not be assessed from the current pose."],
            "suggestions": ["Ensure both shoulders, elbows, and wrists are visible."],
            "metrics": {},
        }

    asymmetry = abs(left_angle - right_angle)
    score = max(0.0, 100.0 - asymmetry * 2.0)

    if asymmetry > 15:
        status = "asymmetric_reach"
        warnings = ["Arm movement is asymmetrical and may indicate compensation."]
        suggestions = [
            "Reduce trunk compensation and aim for equal arm extension on both sides.",
            "Keep the shoulder stable throughout the reach.",
        ]
    elif score >= 80:
        status = "controlled_reach"
        warnings = []
        suggestions = [
            "Maintain smooth elbow extension and balanced shoulder elevation.",
        ]
    else:
        status = "needs_correction"
        warnings = ["The upper limb pattern is less controlled than expected."]
        suggestions = [
            "Shorten the reach range and improve shoulder control before increasing amplitude.",
        ]

    return {
        "status": status,
        "score": round(score, 2),
        "warnings": warnings,
        "suggestions": suggestions,
        "metrics": {
            "left_elbow_angle": round(left_angle, 2),
            "right_elbow_angle": round(right_angle, 2),
            "asymmetry": round(asymmetry, 2),
        },
    }
