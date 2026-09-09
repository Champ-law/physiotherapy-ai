from __future__ import annotations

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - environment-specific guard
    raise ImportError(
        "MediaPipe is not installed correctly. Reinstall the real package: pip install --upgrade mediapipe"
    ) from exc

from src.backbone.base_pose import BasePoseDetector


class PoseDetector(BasePoseDetector):
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        if not hasattr(mp, "solutions"):
            raise ImportError(
                "The installed mediapipe package is incompatible. Install the official MediaPipe package."
            )

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        if frame is None or np.asarray(frame).size == 0:
            return None

        if frame.ndim == 2:
            rgb_frame = frame
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb_frame)
        return results.pose_landmarks if results else None
