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
    from src.backbone.mediapipe_face import FaceDetector, draw_face
    from src.analysis.angle_utils import calculate_angle
    from src.analysis.pose_metrics import (
        ANGLE_TRIPLETS,
        SquatRepTracker,
        TemporalSmoother,
        assess_pose_quality,
        camera_guidance,
        calculate_pose_angles,
    )
    from src.backbone.base_pose import BasePoseDetector
    from src.rules_engine.squat_rules import analyze_squat
except ModuleNotFoundError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.backbone.mediapipe_face import FaceDetector, draw_face
    from src.analysis.angle_utils import calculate_angle
    from src.analysis.pose_metrics import (
        ANGLE_TRIPLETS,
        SquatRepTracker,
        TemporalSmoother,
        assess_pose_quality,
        camera_guidance,
        calculate_pose_angles,
    )
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
        self.last_result = None

    def detect(self, frame):
        if frame is None or np.asarray(frame).size == 0:
            return None

        if frame.ndim == 2:
            rgb_frame = frame
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.last_result = self.pose.process(rgb_frame)
        return self.last_result.pose_landmarks if self.last_result else None

    def detect_with_metadata(self, frame):
        """Return image landmarks, world landmarks, and the raw MediaPipe result."""
        self.detect(frame)
        if self.last_result is None:
            return None
        return {
            "landmarks": self.last_result.pose_landmarks,
            "world_landmarks": self.last_result.pose_world_landmarks,
            "result": self.last_result,
        }


def draw_pose(frame, landmarks):
    connection_pairs = mp.solutions.pose.POSE_CONNECTIONS

    h, w, _ = frame.shape
    for start, end in connection_pairs:
        pt1 = landmarks.landmark[start]
        pt2 = landmarks.landmark[end]
        if min(getattr(pt1, "visibility", 1.0), getattr(pt2, "visibility", 1.0)) < 0.55:
            continue
        x1, y1 = int(pt1.x * w), int(pt1.y * h)
        x2, y2 = int(pt2.x * w), int(pt2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for landmark in landmarks.landmark:
        if getattr(landmark, "visibility", 1.0) < 0.55:
            continue
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        confidence = getattr(landmark, "visibility", 1.0)
        color = (0, 220, 0) if confidence >= 0.8 else (0, 180, 255)
        cv2.circle(frame, (x, y), 4, color, -1)


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
    face_detector = FaceDetector()
    smoother = TemporalSmoother(alpha=0.3)
    rep_tracker = SquatRepTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to access the camera. Check that a webcam is connected and available.")
        raise SystemExit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        metadata = detector.detect_with_metadata(frame)
        face_landmarks = face_detector.detect(frame)
        face_analysis = face_detector.analyze_landmarks(face_landmarks)
        if face_landmarks is not None:
            draw_face(frame, face_landmarks)
            cv2.putText(frame, f"Face: {face_analysis.expression} {face_analysis.confidence:.0%}", (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 180, 0), 2)
        if metadata is not None and metadata["landmarks"] is not None:
            landmarks = metadata["landmarks"]
            draw_pose(frame, landmarks)

            try:
                quality = assess_pose_quality(landmarks)
                angles = calculate_pose_angles(landmarks, world_landmarks=metadata["world_landmarks"])
                smoothed_angles = smoother.update(angles)
                knee_angle = smoothed_angles.get("left_knee_angle")
                rep_data = rep_tracker.update(knee_angle)
                if not quality.reliable or knee_angle is None:
                    raise ValueError("Pose confidence is too low")

                left_hip = get_landmark_xy(landmarks, "left_hip")
                left_knee = get_landmark_xy(landmarks, "left_knee")
                left_ankle = get_landmark_xy(landmarks, "left_ankle")
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

                cv2.putText(frame, f"L knee: {knee_angle:.1f} deg  R knee: {smoothed_angles.get('right_knee_angle', 0):.1f} deg", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                cv2.putText(frame, f"Status: {status_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Phase: {rep_data['phase']}  Reps: {rep_data['repetitions']}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {quality.score:.0f}%", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, suggestion, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception:
                setup_guidance = camera_guidance(landmarks)
                guidance = (quality.guidance + setup_guidance)[0] if quality.guidance or setup_guidance else "Keep your full body visible"
                cv2.putText(frame, guidance, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 255), 2)

        cv2.imshow("Physiotherapy AI - Pose Analysis", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
