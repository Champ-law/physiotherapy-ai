from __future__ import annotations


class BasePoseDetector:
    def detect(self, frame):
        raise NotImplementedError("Subclasses must implement detect(frame).")
