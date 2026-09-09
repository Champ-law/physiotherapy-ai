from __future__ import annotations

import json
from pathlib import Path


def load_exercise_config(exercise_name):
    config_dir = Path(__file__).resolve().parent / "exercise_configs"
    config_file = config_dir / f"{exercise_name}.json"

    if not config_file.exists():
        default = {
            "exercise": exercise_name,
            "thresholds": {
                "good_depth_max": 100,
                "too_deep_min": 85,
                "standing_threshold": 170,
            },
            "feedback": ["Maintain posture."],
        }
        return default

    with config_file.open("r", encoding="utf-8") as f:
        return json.load(f)
