from __future__ import annotations


def validate_analysis_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    exercise = payload.get("exercise", "squat")
    landmarks = payload.get("landmarks")
    side = payload.get("side", "left")

    if not isinstance(landmarks, dict):
        raise ValueError("The 'landmarks' field must be an object mapping names to coordinates.")

    normalized = {
        "exercise": exercise,
        "landmarks": landmarks,
        "side": side,
    }
    return normalized
