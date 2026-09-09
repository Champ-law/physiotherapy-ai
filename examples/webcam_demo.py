# examples/webcam_demo.py
import cv2
from src.backbone.mediapipe_pose import PoseDetector
from src.rules_engine.squat_rules import analyze_squat

detector = PoseDetector()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    landmarks = detector.detect(frame)
    if landmarks:
        feedback = analyze_squat(landmarks.landmark)
        cv2.putText(frame, feedback, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Physiotherapy AI", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
