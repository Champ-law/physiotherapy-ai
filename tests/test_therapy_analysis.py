import numpy as np

from src.analysis.angle_utils import calculate_angle
from src.analysis.face_metrics import analyze_face
from src.analysis.pose_metrics import (
    SquatRepTracker,
    TemporalSmoother,
    assess_pose_quality,
    calculate_pose_angles,
    camera_guidance,
)
from src.analysis.symmetry import analyze_symmetry
from src.api.schemas import validate_analysis_request
from src.backbone.mediapipe_pose import PoseDetector
from src.config.exercise_loader import load_exercise_config
from src.feedback.report import generate_report
from src.models.exercise_result import ExerciseResult, SessionSummary
from src.rules_engine.gait_rules import analyze_gait
from src.rules_engine.mimic_scoring import score_mimic
from src.rules_engine.posture_rules import analyze_posture
from src.rules_engine.squat_rules import analyze_squat
from src.rules_engine.upper_limb_rules import analyze_upper_limb
from src.session.session_manager import SessionManager
from src.video.capture_loop import VideoCaptureLoop
from src.video.live_feedback import LiveFeedbackProcessor


def test_calculate_angle_for_right_triangle():
    angle = calculate_angle((0, 0), (0, 1), (1, 1))
    assert abs(angle - 90.0) < 1e-6


def test_pose_detector_has_detect_method():
    detector = PoseDetector()
    assert hasattr(detector, "detect")


def test_analyze_squat_returns_structured_result():
    landmarks = {
        "left_hip": (0.0, 0.0),
        "left_knee": (0.0, 0.5),
        "left_ankle": (0.0, 1.0),
    }

    result = analyze_squat(landmarks)

    assert "status" in result
    assert "score" in result
    assert "suggestions" in result
    assert isinstance(result["suggestions"], list)


def test_analyze_symmetry_returns_score():
    left = 90.0
    right = 96.0
    result = analyze_symmetry({"left_knee_angle": left, "right_knee_angle": right})

    assert "symmetry_score" in result
    assert 0 <= result["symmetry_score"] <= 100
    assert "warnings" in result


def test_analyze_gait_returns_clinical_summary():
    landmarks = {
        "left_hip": (0.0, 0.5),
        "left_knee": (0.0, 0.2),
        "left_ankle": (0.5, 0.0),
        "right_hip": (2.0, 0.5),
        "right_knee": (2.0, 0.2),
        "right_ankle": (1.5, 0.0),
    }

    result = analyze_gait(landmarks)

    assert "status" in result
    assert "score" in result
    assert "suggestions" in result
    assert isinstance(result["suggestions"], list)


def test_exercise_result_has_unified_schema():
    result = ExerciseResult(
        exercise="squat",
        status="good_depth",
        score=90,
        warnings=["Mild knee drift"],
        suggestions=["Keep knees aligned"],
        metrics={"knee_angle": 92},
    )

    payload = result.to_dict()
    assert payload["exercise"] == "squat"
    assert payload["status"] == "good_depth"
    assert payload["score"] == 90
    assert payload["suggestions"][0] == "Keep knees aligned"


def test_session_summary_aggregates_scores():
    results = [
        ExerciseResult(exercise="squat", status="good_depth", score=90, warnings=[], suggestions=[], metrics={}),
        ExerciseResult(exercise="gait", status="stable_gait", score=80, warnings=[], suggestions=[], metrics={}),
    ]

    summary = SessionSummary(results)

    assert summary.overall_score == 85
    assert len(summary.exercise_results) == 2
    assert summary.to_dict()["overall_score"] == 85


def test_analyze_upper_limb_returns_clinical_summary():
    landmarks = {
        "left_shoulder": (0.0, 0.0),
        "left_elbow": (0.0, 0.4),
        "left_wrist": (0.3, 0.7),
        "right_shoulder": (1.0, 0.0),
        "right_elbow": (1.0, 0.4),
        "right_wrist": (0.7, 0.7),
    }

    result = analyze_upper_limb(landmarks)

    assert "status" in result
    assert "score" in result
    assert "suggestions" in result
    assert isinstance(result["suggestions"], list)


def test_generate_report_creates_pdf_bytes():
    summary = SessionSummary([
        ExerciseResult(exercise="squat", status="good_depth", score=90, warnings=[], suggestions=["Keep knees aligned"], metrics={}),
    ])

    pdf_bytes = generate_report(summary)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_validate_analysis_request_return_schema():
    payload = {
        "exercise": "squat",
        "landmarks": {
            "left_hip": (0.0, 0.0),
            "left_knee": (0.0, 0.5),
            "left_ankle": (0.0, 1.0),
        },
        "side": "left",
    }

    result = validate_analysis_request(payload)

    assert result["exercise"] == "squat"
    assert result["side"] == "left"
    assert "landmarks" in result


def test_exercise_config_loader_loads_thresholds():
    config = load_exercise_config("squat")

    assert isinstance(config, dict)
    assert "thresholds" in config
    assert "good_depth_max" in config["thresholds"]


def test_analyze_posture_returns_summary():
    landmarks = {
        "left_shoulder": (0.1, 0.5),
        "right_shoulder": (0.9, 0.5),
        "left_hip": (0.2, 0.6),
        "right_hip": (0.8, 0.6),
    }

    result = analyze_posture(landmarks)

    assert "status" in result
    assert "score" in result
    assert "suggestions" in result


def test_mimic_scoring_returns_similarity_score():
    actual = {
        "left_shoulder": (0.0, 0.0),
        "left_elbow": (0.1, 0.4),
        "left_wrist": (0.2, 0.8),
    }
    reference = {
        "left_shoulder": (0.0, 0.0),
        "left_elbow": (0.1, 0.5),
        "left_wrist": (0.2, 1.0),
    }

    result = score_mimic(actual, reference)

    assert 0 <= result["score"] <= 100
    assert "similarity" in result


def test_live_feedback_processor_processes_pose():
    processor = LiveFeedbackProcessor(exercise="squat")
    result = processor.analyze_frame({
        "left_hip": (0.0, 0.0),
        "left_knee": (0.0, 0.5),
        "left_ankle": (0.0, 1.0),
    })

    assert "exercise" in result
    assert "status" in result
    assert "score" in result


def test_session_manager_tracks_results_over_time():
    manager = SessionManager()
    manager.add_result({"exercise": "squat", "score": 90})
    manager.add_result({"exercise": "squat", "score": 80})

    assert len(manager.results) == 2
    assert manager.average_score == 85
    assert manager.summary["count"] == 2


def test_capture_loop_has_iteration_and_stop_behavior():
    capture = VideoCaptureLoop(source=0, exercise="squat", frame_limit=1)
    assert hasattr(capture, "run")
    assert hasattr(capture, "stop")
    assert capture.frame_limit == 1


def test_pose_metrics_calculate_bilateral_joint_angles():
    landmarks = {
        "left_shoulder": (0.0, 0.0),
        "right_shoulder": (1.0, 0.0),
        "left_elbow": (0.0, 0.5),
        "right_elbow": (1.0, 0.5),
        "left_wrist": (0.0, 1.0),
        "right_wrist": (1.0, 1.0),
        "left_hip": (0.0, 1.0),
        "right_hip": (1.0, 1.0),
        "left_knee": (0.0, 2.0),
        "right_knee": (1.0, 2.0),
        "left_ankle": (0.0, 3.0),
        "right_ankle": (1.0, 3.0),
        "left_foot_index": (0.0, 4.0),
        "right_foot_index": (1.0, 4.0),
    }

    angles = calculate_pose_angles(landmarks)

    assert angles["left_knee_angle"] == 180.0
    assert angles["right_knee_angle"] == 180.0
    assert "left_elbow_angle" in angles
    assert "torso_angle" in angles


def test_pose_quality_rejects_low_confidence_landmarks():
    landmarks = {
        "left_hip": {"x": 0.0, "y": 0.0, "visibility": 0.2},
        "right_hip": {"x": 1.0, "y": 0.0, "visibility": 0.2},
    }

    quality = assess_pose_quality(landmarks, names=["left_hip", "right_hip"])

    assert not quality.reliable
    assert quality.visible_count == 0
    assert quality.guidance


def test_temporal_smoother_reduces_change_between_values():
    smoother = TemporalSmoother(alpha=0.25)
    smoother.update({"left_knee_angle": 100.0})
    result = smoother.update({"left_knee_angle": 140.0})

    assert result["left_knee_angle"] == 110.0


def test_squat_tracker_counts_completed_rep():
    tracker = SquatRepTracker()

    tracker.update(170)
    tracker.update(100)
    tracker.update(85)
    result = tracker.update(170)

    assert result["repetitions"] == 1
    assert result["phase"] == "standing"


def test_camera_guidance_detects_narrow_or_incomplete_view():
    guidance = camera_guidance({
        "left_shoulder": (0.49, 0.2),
        "right_shoulder": (0.51, 0.2),
        "left_hip": (0.49, 0.5),
        "right_hip": (0.51, 0.5),
        "left_ankle": (0.49, 0.95),
        "right_ankle": (0.51, 0.95),
    })

    assert guidance


def test_face_analysis_returns_orientation_and_expression_metrics():
    face = {
        "nose_tip": (0.5, 0.45, 0.0),
        "left_eye_outer": (0.35, 0.35, 0.0),
        "left_eye_inner": (0.42, 0.35, 0.0),
        "left_eye_top": (0.38, 0.34, 0.0),
        "left_eye_bottom": (0.38, 0.36, 0.0),
        "right_eye_inner": (0.58, 0.35, 0.0),
        "right_eye_outer": (0.65, 0.35, 0.0),
        "right_eye_top": (0.62, 0.34, 0.0),
        "right_eye_bottom": (0.62, 0.36, 0.0),
        "left_cheek": (0.28, 0.5, 0.0),
        "right_cheek": (0.72, 0.5, 0.0),
        "mouth_left": (0.4, 0.62, 0.0),
        "mouth_right": (0.6, 0.62, 0.0),
        "upper_lip": (0.5, 0.61, 0.0),
        "lower_lip": (0.5, 0.64, 0.0),
    }

    result = analyze_face(face)

    assert result.visible
    assert set(result.head_orientation) == {"yaw", "pitch", "roll"}
    assert "left_eye_open" in result.metrics
    assert result.expression in {"neutral-like expression", "smile-like expression", "eyes closed"}
