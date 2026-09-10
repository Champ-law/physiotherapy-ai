from __future__ import annotations

import cv2
import mediapipe as mp

from src.analysis.face_metrics import FACE_INDEXES, analyze_face


class FaceDetector:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if not hasattr(mp, "solutions"):
            raise ImportError("The installed mediapipe package is incompatible with Face Mesh.")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.last_result = None

    def detect(self, frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_result = self.face_mesh.process(rgb_frame)
        if not self.last_result.multi_face_landmarks:
            return None
        return self.last_result.multi_face_landmarks[0]

    def analyze(self, frame):
        landmarks = self.detect(frame)
        return self.analyze_landmarks(landmarks)

    @staticmethod
    def analyze_landmarks(landmarks):
        return analyze_face(landmarks) if landmarks is not None else analyze_face({})


def draw_face(frame, landmarks, color=(255, 180, 0)):
    if landmarks is None:
        return
    height, width = frame.shape[:2]
    for point in landmarks.landmark:
        x, y = int(point.x * width), int(point.y * height)
        cv2.circle(frame, (x, y), 1, color, -1)

    analysis = analyze_face(landmarks)
    if analysis.bounding_box:
        left, top, right, bottom = analysis.bounding_box
        cv2.rectangle(frame, (int(left * width), int(top * height)), (int(right * width), int(bottom * height)), color, 1)