# src/rules_engine/squat_rules.py
from __future__ import annotations

from src.analysis.angle_utils import calculate_angle, get_landmark_xy
from src.config.thresholds import SQUAT_THRESHOLDS


def _resolve_side_landmarks(landmarks, side):
    side = side.lower()
    hip_name = f"{side}_hip"
    knee_name = f"{side}_knee"
    ankle_name = f"{side}_ankle"

    return (
        get_landmark_xy(landmarks, hip_name),
        get_landmark_xy(landmarks, knee_name),
        get_landmark_xy(landmarks, ankle_name),
    )


def analyze_squat(landmarks, side="left"):
    """Return a structured squat analysis result."""
    try:
        hip, knee, ankle = _resolve_side_landmarks(landmarks, side)
        knee_angle = calculate_angle(hip, knee, ankle)
    except (KeyError, TypeError, ValueError):
        return {
            "status": "invalid_pose",
            "score": 0,
            "angle": None,
            "warnings": ["A valid squat pose could not be detected."],
            "suggestions": ["Ensure the full body is visible to the camera."],
            "metrics": {},
        }

    thresholds = SQUAT_THRESHOLDS

    if knee_angle < thresholds["too_deep_min"]:
        status = "too_deep"
        score = 60
        warnings = ["Knee bend exceeds the recommended squat depth."]
        suggestions = ["Reduce the depth slightly and keep the chest tall."]
    elif knee_angle <= thresholds["good_depth_max"]:
        status = "good_depth"
        score = 90
        warnings = []
        suggestions = ["Maintain this depth and keep the knees aligned with the toes."]
    elif knee_angle < thresholds["standing_threshold"]:
        status = "descending"
        score = 75
        warnings = ["The squat is in progress. Keep movement controlled."]
        suggestions = ["Lower the hips smoothly while keeping the torso upright."]
    else:
        status = "standing"
        score = 25
        warnings = ["The body is nearly standing. Increase the squat range."]
        suggestions = ["Begin lowering the hips and maintain a stable trunk position."]

    return {
        "status": status,
        "score": score,
        "angle": round(knee_angle, 2),
        "warnings": warnings,
        "suggestions": suggestions,
        "metrics": {
            "knee_angle": round(knee_angle, 2),
            "side": side,
            "depth_score": score / 100,
        },
    }
