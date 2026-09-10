from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from src.analysis.angle_utils import calculate_angle


POSE_INDEXES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

ANGLE_TRIPLETS = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_ankle_angle": ("left_knee", "left_ankle", "left_foot_index"),
    "right_ankle_angle": ("right_knee", "right_ankle", "right_foot_index"),
    "torso_angle": ("left_shoulder", "left_hip", "left_knee"),
}


@dataclass
class PoseQuality:
    visibility: Dict[str, float]
    visible_count: int
    total_count: int
    guidance: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.total_count:
            return 0.0
        return round(sum(self.visibility.values()) / self.total_count * 100, 1)

    @property
    def reliable(self) -> bool:
        return self.visible_count >= max(4, int(self.total_count * 0.7))


class TemporalSmoother:
    """Exponential moving average for noisy coordinates or angle values."""

    def __init__(self, alpha: float = 0.35):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be greater than 0 and no greater than 1")
        self.alpha = alpha
        self._values: Dict[str, np.ndarray | float] = {}

    def update(self, values: Mapping[str, Any]) -> Dict[str, Any]:
        smoothed: Dict[str, Any] = {}
        for name, value in values.items():
            current = np.asarray(value, dtype=float) if isinstance(value, (list, tuple, np.ndarray)) else float(value)
            previous = self._values.get(name)
            if previous is None:
                result = current
            else:
                result = self.alpha * current + (1 - self.alpha) * previous
            self._values[name] = result
            smoothed[name] = result.copy() if isinstance(result, np.ndarray) else float(result)
        return smoothed

    def reset(self):
        self._values.clear()


def _landmark_value(landmarks: Any, name: str, world: bool = False):
    if isinstance(landmarks, Mapping):
        value = landmarks.get(name)
        if value is None:
            raise KeyError(name)
        array = np.asarray(value[:3] if world else value[:2], dtype=float)
        if array.size < (3 if world else 2):
            raise ValueError(f"Landmark '{name}' has insufficient coordinates")
        return array

    if not hasattr(landmarks, "landmark"):
        raise TypeError("Expected a landmark mapping or MediaPipe landmark collection")
    index = POSE_INDEXES[name]
    point = landmarks.landmark[index]
    coordinates = (point.x, point.y, getattr(point, "z", 0.0)) if world else (point.x, point.y)
    return np.asarray(coordinates, dtype=float)


def _visibility_value(landmarks: Any, name: str) -> float:
    if isinstance(landmarks, Mapping):
        value = landmarks.get(name)
        if isinstance(value, Mapping):
            return float(value.get("visibility", value.get("confidence", 1.0)))
        return 1.0 if value is not None else 0.0
    if not hasattr(landmarks, "landmark"):
        return 0.0
    return float(np.clip(getattr(landmarks.landmark[POSE_INDEXES[name]], "visibility", 1.0), 0.0, 1.0))


def extract_pose_points(landmarks: Any, names: Optional[Iterable[str]] = None, min_visibility: float = 0.55):
    names = list(names or POSE_INDEXES)
    points = {}
    visibility = {}
    for name in names:
        confidence = _visibility_value(landmarks, name)
        visibility[name] = confidence
        if confidence >= min_visibility:
            points[name] = _landmark_value(landmarks, name)
    return points, visibility


def assess_pose_quality(landmarks: Any, names: Optional[Iterable[str]] = None, min_visibility: float = 0.55) -> PoseQuality:
    selected_names = list(names or POSE_INDEXES)
    _, visibility = extract_pose_points(landmarks, selected_names, min_visibility)
    visible_count = sum(confidence >= min_visibility for confidence in visibility.values())
    guidance = []
    if visible_count == 0:
        guidance.append("No pose detected - step into view")
    elif visible_count < max(4, int(len(selected_names) * 0.7)):
        guidance.append("Keep your full body visible")
    if visibility.get("left_ankle", 1.0) < min_visibility or visibility.get("right_ankle", 1.0) < min_visibility:
        guidance.append("Move back so both feet are visible")
    if visibility.get("left_shoulder", 1.0) < min_visibility or visibility.get("right_shoulder", 1.0) < min_visibility:
        guidance.append("Face the camera and keep both shoulders visible")
    return PoseQuality(visibility, visible_count, len(selected_names), guidance)


def camera_guidance(landmarks: Any, min_visibility: float = 0.55) -> list[str]:
    """Return practical setup guidance based on framing and visible landmarks."""
    points, _ = extract_pose_points(landmarks, min_visibility=min_visibility)
    guidance = []
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle")
    if not all(name in points for name in required):
        return ["Use a side or front view and keep your entire body in frame"]

    body_points = np.asarray(list(points.values()), dtype=float)
    x_min, y_min = body_points[:, 0].min(), body_points[:, 1].min()
    x_max, y_max = body_points[:, 0].max(), body_points[:, 1].max()
    if x_min < 0.05 or x_max > 0.95 or y_min < 0.05 or y_max > 0.95:
        guidance.append("Move farther back so your body has space around it")

    shoulder_width = abs(points["left_shoulder"][0] - points["right_shoulder"][0])
    hip_width = abs(points["left_hip"][0] - points["right_hip"][0])
    if shoulder_width < 0.08 and hip_width < 0.08:
        guidance.append("Turn more side-on or face the camera clearly")
    return guidance


def calculate_pose_angles(landmarks: Any, min_visibility: float = 0.55, world_landmarks: Any = None):
    """Calculate reliable bilateral joint angles, preferring world coordinates when available."""
    points, visibility = extract_pose_points(landmarks, min_visibility=min_visibility)
    world_points = {}
    if world_landmarks is not None:
        world_points, _ = extract_pose_points(world_landmarks, min_visibility=min_visibility)

    angles = {}
    for angle_name, (first, vertex, last) in ANGLE_TRIPLETS.items():
        source = world_points if all(name in world_points for name in (first, vertex, last)) else points
        if all(name in source for name in (first, vertex, last)):
            try:
                angles[angle_name] = round(calculate_angle(source[first], source[vertex], source[last]), 2)
            except (TypeError, ValueError):
                continue
    return angles


@dataclass
class SquatRepTracker:
    """Track squat movement phases without counting partial or noisy cycles."""

    phase: str = "standing"
    repetitions: int = 0
    min_angle: Optional[float] = None

    def update(self, knee_angle: Optional[float], standing_threshold: float = 160, depth_threshold: float = 110):
        if knee_angle is None:
            return {"phase": "unknown", "repetitions": self.repetitions, "min_angle": self.min_angle}

        if self.min_angle is None or knee_angle < self.min_angle:
            self.min_angle = round(knee_angle, 2)

        if knee_angle >= standing_threshold:
            if self.phase in {"bottom", "ascending"}:
                self.repetitions += 1
            self.phase = "standing"
        elif knee_angle <= depth_threshold:
            self.phase = "bottom"
        elif self.phase == "bottom" or self.phase == "descending":
            self.phase = "ascending"
        else:
            self.phase = "descending"

        if self.phase == "standing":
            self.min_angle = None
        return {"phase": self.phase, "repetitions": self.repetitions, "min_angle": self.min_angle}