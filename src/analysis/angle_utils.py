# src/analysis/angle_utils.py
from __future__ import annotations

import numpy as np


def _as_point(value):
    if isinstance(value, np.ndarray):
        point = value.astype(float)
    elif isinstance(value, (list, tuple)):
        point = np.asarray(value, dtype=float)
    else:
        raise TypeError("Landmark must be a coordinate tuple/list/array.")

    if point.size < 2:
        raise ValueError("Each landmark must contain at least x and y coordinates.")

    return point[:2]


def calculate_angle(a, b, c):
    """Return the angle in degrees between vectors BA and BC."""
    a = _as_point(a)
    b = _as_point(b)
    c = _as_point(c)

    ba = a - b
    bc = c - b

    if np.allclose(ba, 0) or np.allclose(bc, 0):
        raise ValueError("Degenerate angle: the joint point is identical to one end point.")

    cosine = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return float(np.degrees(np.arccos(cosine)))


def get_landmark_xy(landmarks, landmark_name):
    """Resolve a named landmark from either a dict or a MediaPipe landmarks object."""
    if isinstance(landmarks, dict):
        normalized_name = landmark_name.lower().replace("-", "_")
        candidates = [landmark_name, normalized_name]
        for candidate in candidates:
            if candidate in landmarks:
                return np.asarray(landmarks[candidate][:2], dtype=float)
        raise KeyError(f"Landmark '{landmark_name}' was not found in the landmark dictionary.")

    if hasattr(landmarks, "landmark"):
        landmark = landmarks.landmark[landmark_name]
        return np.asarray([landmark.x, landmark.y], dtype=float)

    raise TypeError("Unsupported landmark representation. Expected dict or MediaPipe landmark object.")
