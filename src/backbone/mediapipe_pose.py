from __future__ import annotations

import os
import sys

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - environment-specific guard
    raise ImportError(
        "MediaPipe is not installed correctly. Reinstall the real package: pip install --upgrade mediapipe"
    ) from exc

try:
    from src.analysis.angle_utils import calculate_angle
    from src.backbone.base_pose import BasePoseDetector
    from src.rules_engine.squat_rules import analyze_squat
except ModuleNotFoundError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.analysis.angle_utils import calculate_angle
    from src.backbone.base_pose import BasePoseDetector
    from src.rules_engine.squat_rules import analyze_squat


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


def draw_pose(frame, landmarks):
    connection_pairs = [
        (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (23, 11), (24, 12), (23, 27), (24, 28)
    ]

    h, w, _ = frame.shape
    for start, end in connection_pairs:
        pt1 = landmarks.landmark[start]
        pt2 = landmarks.landmark[end]
        x1, y1 = int(pt1.x * w), int(pt1.y * h)
        x2, y2 = int(pt2.x * w), int(pt2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for landmark in landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)


def get_landmark_xy(landmarks, landmark_name):
    if isinstance(landmark_name, (int, np.integer)):
        idx = int(landmark_name)
        if not hasattr(landmarks, "landmark"):
            raise ValueError("Expected a MediaPipe landmark object.")
        try:
            landmark = landmarks.landmark[idx]
        except IndexError as exc:
            raise KeyError(f"Landmark index '{idx}' was not found in the MediaPipe landmark set.") from exc
        return np.asarray([landmark.x, landmark.y], dtype=float)

    if hasattr(landmarks, "landmark"):
        normalized_name = str(landmark_name).lower().replace("-", "_")
        lookup = {
            "left_hip": 23,
            "left_knee": 25,
            "left_ankle": 27,
            "right_hip": 24,
            "right_knee": 26,
            "right_ankle": 28,
        }
        if normalized_name in lookup:
            return get_landmark_xy(landmarks, lookup[normalized_name])
        raise KeyError(f"Landmark '{landmark_name}' was not found in the MediaPipe landmark set.")

    raise ValueError("Expected a MediaPipe landmark object.")


if __name__ == "__main__":
    print("Starting MediaPipe webcam demo. Press 'q' to quit.")
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to access the camera. Check that a webcam is connected and available.")
        raise SystemExit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        landmarks = detector.detect(frame)
        if landmarks is not None:
            draw_pose(frame, landmarks)

            try:
                left_hip = get_landmark_xy(landmarks, "left_hip")
                left_knee = get_landmark_xy(landmarks, "left_knee")
                left_ankle = get_landmark_xy(landmarks, "left_ankle")
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                feedback = analyze_squat({
                    "left_hip": left_hip,
                    "left_knee": left_knee,
                    "left_ankle": left_ankle,
                })
                status = feedback.get("status", "analyzing")
                status_text = {
                    "too_deep": "Too deep",
                    "good_depth": "Good depth",
                    "descending": "Descending",
                    "standing": "Standing",
                    "invalid_pose": "Pose not clear",
                }.get(status, status.replace("_", " ").title())
                suggestion = feedback.get("suggestions", ["Keep moving smoothly"])[0]

                cv2.putText(frame, f"Knee angle: {knee_angle:.1f}°", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Status: {status_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, suggestion, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            except Exception:
                cv2.putText(frame, "Pose not fully visible", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Physiotherapy AI - Pose Analysis", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
