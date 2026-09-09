from __future__ import annotations

from src.analysis.angle_utils import get_landmark_xy


def _average_xy(landmarks, key_a, key_b):
    a = get_landmark_xy(landmarks, key_a)
    b = get_landmark_xy(landmarks, key_b)
    return (a + b) / 2.0


def analyze_posture(landmarks):
    """Assess frontal alignment based on shoulder and hip symmetry."""
    try:
        left_shoulder = get_landmark_xy(landmarks, "left_shoulder")
        right_shoulder = get_landmark_xy(landmarks, "right_shoulder")
        left_hip = get_landmark_xy(landmarks, "left_hip")
        right_hip = get_landmark_xy(landmarks, "right_hip")
    except (KeyError, TypeError, ValueError):
        return {
            "status": "invalid_pose",
            "score": 0,
            "warnings": ["Posture landmarks are missing."],
            "suggestions": ["Ensure shoulders and hips are visible in the camera frame."],
            "metrics": {},
        }

    shoulder_gap = abs(left_shoulder[0] - right_shoulder[0])
    hip_gap = abs(left_hip[0] - right_hip[0])
    alignment_error = abs(shoulder_gap - hip_gap)
    score = max(0.0, 100.0 - alignment_error * 100.0)

    if alignment_error > 0.12:
        status = "postural_misalignment"
        warnings = ["Shoulder and hip alignment is uneven."]
        suggestions = ["Promote a vertical trunk line and keep the pelvis centered."]
    elif score >= 80:
        status = "good_posture"
        warnings = []
        suggestions = ["Maintain this neutral alignment throughout the exercise."]
    else:
        status = "needs_correction"
        warnings = ["Posture is drifting away from neutral alignment."]
        suggestions = ["Reduce trunk lean and improve pelvic symmetry."]

    return {
        "status": status,
        "score": round(score, 2),
        "warnings": warnings,
        "suggestions": suggestions,
        "metrics": {
            "shoulder_gap": round(float(shoulder_gap), 4),
            "hip_gap": round(float(hip_gap), 4),
            "alignment_error": round(float(alignment_error), 4),
        },
    }
