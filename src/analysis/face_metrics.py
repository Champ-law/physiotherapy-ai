from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


FACE_INDEXES = {
    "nose_tip": 1,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "left_cheek": 234,
    "right_cheek": 454,
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip": 13,
    "lower_lip": 14,
}


@dataclass
class FaceAnalysis:
    visible: bool
    confidence: float
    bounding_box: tuple[float, float, float, float] | None = None
    head_orientation: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    expression: str = "unavailable"
    guidance: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "visible": self.visible,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "head_orientation": self.head_orientation,
            "metrics": self.metrics,
            "expression": self.expression,
            "guidance": self.guidance,
        }


def _point(face: Any, name: str) -> np.ndarray:
    if isinstance(face, Mapping):
        value = face[name]
        if isinstance(value, Mapping):
            value = (value["x"], value["y"], value.get("z", 0.0))
        return np.asarray(value[:3], dtype=float)
    if not hasattr(face, "landmark"):
        raise TypeError("Expected a face mapping or MediaPipe face landmark collection")
    point = face.landmark[FACE_INDEXES[name]]
    return np.asarray([point.x, point.y, getattr(point, "z", 0.0)], dtype=float)


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return 0.0
    return float(numerator / denominator)


def analyze_face(face: Any, min_confidence: float = 0.5) -> FaceAnalysis:
    """Estimate face geometry and visible expression cues, without diagnosing emotion."""
    try:
        points = {name: _point(face, name) for name in FACE_INDEXES}
    except (KeyError, TypeError, ValueError):
        return FaceAnalysis(False, 0.0, guidance=["Face not visible - look toward the camera"])

    xs = np.asarray([point[0] for point in points.values()])
    ys = np.asarray([point[1] for point in points.values()])
    box = (round(float(xs.min()), 4), round(float(ys.min()), 4), round(float(xs.max()), 4), round(float(ys.max()), 4))

    eye_mid_left = (points["left_eye_top"] + points["left_eye_bottom"]) / 2
    eye_mid_right = (points["right_eye_top"] + points["right_eye_bottom"]) / 2
    eye_line = points["right_eye_outer"] - points["left_eye_outer"]
    eye_distance = np.linalg.norm(points["right_eye_outer"] - points["left_eye_outer"])
    face_width = np.linalg.norm(points["right_cheek"] - points["left_cheek"])
    eye_open_left = _ratio(np.linalg.norm(points["left_eye_top"] - points["left_eye_bottom"]), eye_distance)
    eye_open_right = _ratio(np.linalg.norm(points["right_eye_top"] - points["right_eye_bottom"]), eye_distance)
    mouth_width = np.linalg.norm(points["mouth_right"] - points["mouth_left"])
    mouth_open = _ratio(np.linalg.norm(points["upper_lip"] - points["lower_lip"]), mouth_width)
    mouth_width_ratio = _ratio(mouth_width, face_width)

    eye_midpoint = (eye_mid_left + eye_mid_right) / 2
    cheek_midpoint = (points["left_cheek"] + points["right_cheek"]) / 2
    nose_offset = _ratio(float(points["nose_tip"][0] - cheek_midpoint[0]), face_width)
    roll = float(np.degrees(np.arctan2(eye_line[1], eye_line[0])))
    yaw = float(np.clip(nose_offset * 180, -90, 90))
    pitch = float(np.clip((points["nose_tip"][1] - eye_midpoint[1]) / max(face_width, 1e-6) * 90, -90, 90))

    metrics = {
        "left_eye_open": round(eye_open_left, 4),
        "right_eye_open": round(eye_open_right, 4),
        "mouth_open": round(mouth_open, 4),
        "mouth_width_ratio": round(mouth_width_ratio, 4),
    }
    expression = "eyes closed" if eye_open_left < 0.16 and eye_open_right < 0.16 else "smile-like expression" if mouth_width_ratio > 0.42 and mouth_open < 0.45 else "neutral-like expression"
    guidance = []
    if face_width < 0.08:
        guidance.append("Move closer so facial features are visible")
    if abs(yaw) > 35:
        guidance.append("Turn your face more toward the camera")
    confidence = float(np.clip(min(1.0, face_width * 8), 0.0, 1.0))
    return FaceAnalysis(
        visible=confidence >= min_confidence,
        confidence=round(confidence, 3),
        bounding_box=box,
        head_orientation={"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)},
        metrics=metrics,
        expression=expression,
        guidance=guidance,
    )