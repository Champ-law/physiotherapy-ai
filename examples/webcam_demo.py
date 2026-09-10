# examples/webcam_demo.py
import cv2

from src.analysis.pose_metrics import SquatRepTracker, TemporalSmoother, assess_pose_quality, calculate_pose_angles, camera_guidance
from src.backbone.mediapipe_face import FaceDetector, draw_face
from src.backbone.mediapipe_pose import PoseDetector, draw_pose
from src.rules_engine.squat_rules import analyze_squat


def main():
    detector = PoseDetector()
    face_detector = FaceDetector()
    smoother = TemporalSmoother(alpha=0.3)
    rep_tracker = SquatRepTracker()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        metadata = detector.detect_with_metadata(frame)
        face_landmarks = face_detector.detect(frame)
        face_analysis = face_detector.analyze_landmarks(face_landmarks)
        if face_landmarks:
            draw_face(frame, face_landmarks)
            cv2.putText(
                frame,
                f"Face: {face_analysis.expression} | {face_analysis.confidence:.0%}",
                (20, frame.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 180, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Head Y/P/R: {face_analysis.head_orientation.get('yaw', 0):.0f}/{face_analysis.head_orientation.get('pitch', 0):.0f}/{face_analysis.head_orientation.get('roll', 0):.0f}",
                (20, frame.shape[0] - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 180, 0),
                2,
            )
        else:
            cv2.putText(frame, "Face not visible", (20, frame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)

        if metadata and metadata["landmarks"]:
            landmarks = metadata["landmarks"]
            draw_pose(frame, landmarks)
            quality = assess_pose_quality(landmarks)
            angles = calculate_pose_angles(landmarks, world_landmarks=metadata["world_landmarks"])
            smoothed = smoother.update(angles)
            knee_angle = smoothed.get("left_knee_angle")
            rep_data = rep_tracker.update(knee_angle)

            if quality.reliable and knee_angle is not None:
                feedback = analyze_squat(landmarks)
                message = feedback.get("status", "analyzing").replace("_", " ").title()
                score = feedback.get("score", 0)
                suggestion = feedback.get("suggestions", ["Move smoothly"])[0]
                color = (0, 220, 0) if message == "Good Depth" else (0, 200, 255)
                cv2.putText(frame, f"Status: {message} | Score: {score}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.putText(frame, f"L knee: {knee_angle:.1f} deg | R knee: {smoothed.get('right_knee_angle', 0):.1f} deg", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, f"Phase: {rep_data['phase']} | Reps: {rep_data['repetitions']} | Confidence: {quality.score:.0f}%", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
                cv2.putText(frame, suggestion, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
            else:
                setup_guidance = camera_guidance(landmarks)
                guidance = (quality.guidance + setup_guidance)[0] if quality.guidance or setup_guidance else "Keep your full body visible"
                cv2.putText(frame, guidance, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 255), 2)
                cv2.putText(frame, f"Confidence: {quality.score:.0f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Step into view", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)

        cv2.imshow("Physiotherapy AI - Calibrated Pose Analysis", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
