# Physiotherapy AI

A modular computer-vision-based physiotherapy analysis engine for movement tracking, exercise scoring, and rehabilitation feedback.

## Features

- MediaPipe-based pose detection
- Joint angle analysis
- Squat, gait, posture, and upper-limb assessment
- Symmetry and mimic scoring
- Structured clinical result objects
- PDF session summaries
- Live feedback processing and session tracking

## Project structure

- `src/backbone` — pose tracking adapters
- `src/analysis` — geometry and motion analysis
- `src/rules_engine` — exercise-specific logic
- `src/feedback` — feedback and reporting
- `src/api` — request validation and API layer
- `src/session` — session history tracking
- `src/video` — webcam/live processing
- `src/config` — exercise parameter files
- `tests` — regression tests

## Setup

```bash
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest -q
```

## Example usage

```python
from src.rules_engine.squat_rules import analyze_squat

landmarks = {
    "left_hip": (0.0, 0.0),
    "left_knee": (0.0, 0.5),
    "left_ankle": (0.0, 1.0),
}

result = analyze_squat(landmarks)
print(result)
```

## Notes

This project is designed as a modular foundation for future clinical and API extensions.
