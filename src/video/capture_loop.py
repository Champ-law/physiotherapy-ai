from __future__ import annotations

import time

import cv2

from src.video.live_feedback import LiveFeedbackProcessor


class VideoCaptureLoop:
    def __init__(self, source=0, exercise="squat", frame_limit=None, interval=0.1):
        self.source = source
        self.exercise = exercise
        self.frame_limit = frame_limit
        self.interval = interval
        self.processor = LiveFeedbackProcessor(exercise=exercise)
        self.active = False
        self.frames_processed = 0

    def run(self, callback=None):
        self.active = True
        cap = cv2.VideoCapture(self.source)

        try:
            while self.active and (self.frame_limit is None or self.frames_processed < self.frame_limit):
                ok, frame = cap.read()
                if not ok:
                    break

                self.frames_processed += 1
                if callback:
                    callback(frame)

                result = self.processor.analyze_frame({
                    "left_hip": (0.0, 0.0),
                    "left_knee": (0.0, 0.5),
                    "left_ankle": (0.0, 1.0),
                })
                if callback:
                    callback(result)

                time.sleep(self.interval)
        finally:
            cap.release()

    def stop(self):
        self.active = False
