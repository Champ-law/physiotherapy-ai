from __future__ import annotations

from src.rules_engine.squat_rules import analyze_squat
from src.analysis.pose_metrics import SquatRepTracker, TemporalSmoother, assess_pose_quality, calculate_pose_angles, camera_guidance


class LiveFeedbackProcessor:
    def __init__(self, exercise="squat"):
        self.exercise = exercise
        self.smoother = TemporalSmoother(alpha=0.3)
        self.rep_tracker = SquatRepTracker()

    def analyze_frame(self, landmarks, world_landmarks=None):
        quality = assess_pose_quality(landmarks)
        angles = self.smoother.update(calculate_pose_angles(landmarks, world_landmarks=world_landmarks))
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
            "metrics": {
                **result.get("metrics", {}),
                **angles,
                "pose_confidence": quality.score,
                "pose_reliable": quality.reliable,
                **self.rep_tracker.update(angles.get("left_knee_angle")),
            },
            "guidance": quality.guidance + camera_guidance(landmarks),
        }
