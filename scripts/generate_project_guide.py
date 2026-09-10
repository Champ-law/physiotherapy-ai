from __future__ import annotations

import os
import sys
from pathlib import Path
from textwrap import wrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "physiotherapy_ai_project_guide.pdf"


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="GuideTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=29,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#12304A"),
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="GuideSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#536777"),
        spaceAfter=24,
    ))
    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#12304A"),
        spaceBefore=12,
        spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        name="Subsection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1D6072"),
        spaceBefore=8,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="BodyGuide",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#263746"),
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="SmallGuide",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#536777"),
    ))
    return styles


def bullet(text, styles):
    return Paragraph(f"&#8226; {text}", styles["BodyGuide"])


def section(title, paragraphs, styles, bullets=()):
    flow = [Paragraph(title, styles["Section"])]
    flow.extend(Paragraph(item, styles["BodyGuide"]) for item in paragraphs)
    flow.extend(bullet(item, styles) for item in bullets)
    return flow


def draw_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E1E7"))
    canvas.line(0.65 * inch, 0.55 * inch, 7.85 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#718391"))
    canvas.drawString(0.65 * inch, 0.35 * inch, "Physiotherapy AI project guide")
    canvas.drawRightString(7.85 * inch, 0.35 * inch, f"Page {document.page}")
    canvas.restoreState()


def create_guide(output_path=OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.75 * inch,
        title="Physiotherapy AI Project Guide",
        author="Physiotherapy AI",
    )

    story = [
        Spacer(1, 0.65 * inch),
        Paragraph("Physiotherapy AI", styles["GuideTitle"]),
        Paragraph("How the computer-vision analysis engine works", styles["GuideSubtitle"]),
        Paragraph(
            "This guide explains how the project receives visual input, detects body landmarks, converts those landmarks into measurements, interprets movement, tracks results, and presents live feedback.",
            styles["BodyGuide"],
        ),
        Spacer(1, 0.2 * inch),
    ]

    overview_data = [
        ["Layer", "Responsibility", "Current implementation"],
        ["Visual input", "Capture frames from a webcam", "OpenCV"],
        ["Machine learning", "Estimate body landmarks from pixels", "MediaPipe Pose"],
        ["Geometry", "Calculate joint angles and coordinates", "NumPy and angle utilities"],
        ["Interpretation", "Classify movement and generate advice", "Rule-based exercise engine"],
        ["Tracking", "Collect results and calculate summaries", "Session manager and models"],
        ["Presentation", "Show overlays, API responses, and PDFs", "OpenCV, Flask, ReportLab"],
    ]
    overview_table = Table(overview_data, colWidths=[1.15 * inch, 2.85 * inch, 2.7 * inch], repeatRows=1)
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D6DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([Paragraph("System at a glance", styles["Section"]), overview_table, PageBreak()])

    story.extend(section(
        "1. What makes this a machine-learning project?",
        [
            "The machine-learning part is MediaPipe Pose. It contains a pretrained neural network that has learned visual patterns associated with human body structure. Given an image frame, it estimates the location of body landmarks such as shoulders, hips, knees, ankles, elbows, and wrists.",
            "The project does not train a new model during each webcam session. Instead, it uses the pretrained model for inference: pixels go in, landmark coordinates come out. This is still a machine-learning application because the central visual perception step is performed by a trained neural network.",
            "The rest of the project is an engineering and clinical-logic layer around that model. It makes the raw predictions useful, explainable, and specific to exercises such as squats.",
        ],
        styles,
        bullets=[
            "Learned component: MediaPipe Pose landmark estimation.",
            "Deterministic components: angle calculations, thresholds, scores, status labels, and feedback text.",
            "Application goal: transform visual pose estimates into understandable movement guidance.",
        ],
    ))

    story.extend(section(
        "2. From camera pixels to body landmarks",
        [
            "OpenCV reads a frame from the webcam. The frame is in BGR color format, so the pose detector converts it to RGB before passing it to MediaPipe.",
            "MediaPipe processes the frame and returns a pose landmark collection. Each landmark contains normalized coordinates, generally represented relative to the frame width and height, along with visibility information in the underlying result.",
            "The live demo draws selected landmark connections as a green skeleton and marks detected points with blue dots. This overlay is a visual explanation of what the model is currently using as the body representation.",
        ],
        styles,
        bullets=[
            "Source: src/backbone/mediapipe_pose.py",
            "Input: a live camera frame.",
            "Output: a set of estimated body landmark coordinates.",
            "Important limitation: the current measurements are primarily 2D image-space estimates.",
        ],
    ))

    story.extend(section(
        "3. How movement is measured",
        [
            "The analysis layer resolves named landmarks such as left_hip, left_knee, and left_ankle. The angle utility treats the knee as the vertex and calculates the angle between the hip-to-knee and ankle-to-knee vectors.",
            "For a squat, the knee angle is used as a compact measurement of leg bend. The value changes as the person moves from standing into a squat and back upward.",
            "This is a geometric measurement, not a direct measurement from a physical goniometer. Camera placement, perspective, landmark quality, and occlusion can change the estimate.",
        ],
        styles,
        bullets=[
            "Source: src/analysis/angle_utils.py",
            "Example measurement: left hip -> left knee -> left ankle.",
            "Output: an angle in degrees and related metric values.",
        ],
    ))

    story.extend(section(
        "4. How the project interprets a squat",
        [
            "The squat rules engine receives the landmark coordinates, calculates the knee angle, and compares it with configured thresholds. It then returns a structured result containing status, score, warnings, suggestions, and metrics.",
            "The current squat interpretation distinguishes between standing, descending, good depth, and too deep. These labels are rule-based interpretations of the estimated angle; they are not produced directly by the neural network.",
            "For example, a knee angle between the configured good-depth limits receives the good_depth status and a high score. A very small angle is classified as too_deep, while a large angle is classified as standing.",
        ],
        styles,
        bullets=[
            "Source: src/rules_engine/squat_rules.py",
            "Threshold configuration: src/config/thresholds.py and exercise configuration files.",
            "Result format: status, score, warnings, suggestions, and metrics.",
        ],
    ))

    story.extend(section(
        "5. How live feedback is presented",
        [
            "For each camera frame where a usable pose is found, the live demo draws the skeleton, displays the knee angle, converts the rule status into readable text, and shows the first coaching suggestion.",
            "The display updates continuously as new frames arrive. In that sense, the figures and interpretations are real-time estimates: they represent the most recent processed frame rather than a manually entered or pre-recorded value.",
            "The live feedback processor provides the same exercise-analysis concept as a reusable Python component, while the Flask API exposes structured analysis for other applications.",
        ],
        styles,
        bullets=[
            "Webcam presentation: src/backbone/mediapipe_pose.py and examples/webcam_demo.py.",
            "Reusable streaming logic: src/video/live_feedback.py.",
            "API endpoint: POST /analyze_pose.",
        ],
    ))

    story.extend(section(
        "6. How movement data is tracked",
        [
            "An individual frame produces one analysis result. The session layer can collect multiple results over time, preserve the exercise status and score, and calculate an average score for the session.",
            "The data model gives each result a consistent structure. This makes it possible to store warnings, suggestions, and metrics alongside the exercise name and score rather than losing them in display-only text.",
            "The current session manager is lightweight and in-memory. A production version could add timestamps, repetition detection, confidence filtering, persistent storage, user accounts, and clinician review workflows.",
        ],
        styles,
        bullets=[
            "Session collection: src/session/session_manager.py.",
            "Structured results: src/models/exercise_result.py.",
            "PDF session summaries: src/feedback/report.py.",
        ],
    ))

    story.append(PageBreak())
    story.extend(section(
        "7. Project components",
        [
            "The repository is organized so that visual detection, mathematical analysis, exercise interpretation, presentation, and reporting can evolve independently.",
        ],
        styles,
        bullets=[
            "backbone: connects the application to MediaPipe pose detection.",
            "analysis: converts landmarks into geometric measurements.",
            "rules_engine: interprets measurements for specific exercises.",
            "config: stores exercise thresholds and parameters.",
            "video: supports live frame processing and feedback.",
            "api: validates requests and returns structured JSON results.",
            "models: defines consistent exercise and session result structures.",
            "session: aggregates results over a therapy session.",
            "feedback: produces readable advice and PDF reports.",
            "tests: verifies geometry, rules, schemas, sessions, reports, and integration behavior.",
        ],
    ))

    story.extend(section(
        "8. How to run it",
        [
            "From PowerShell in the project directory, run the webcam demo with the project virtual environment. Press q in the webcam window to stop it.",
        ],
        styles,
        bullets=[
            "Webcam demo: .\\.venv\\Scripts\\python.exe .\\src\\backbone\\mediapipe_pose.py",
            "Tests: .\\.venv\\Scripts\\python.exe -m pytest -q",
            "API server: .\\.venv\\Scripts\\python.exe .\\src\\api\\server.py",
        ],
    ))

    story.extend(section(
        "9. Accuracy, safety, and interpretation",
        [
            "The system is a prototype analysis and feedback tool. It should not be treated as a medical diagnosis system or as a replacement for a qualified physiotherapist.",
            "The quality of results depends on camera position, lighting, body visibility, clothing, movement speed, landmark confidence, and whether the chosen body side is clearly visible.",
            "The displayed score is a software-defined exercise score based on current thresholds. It is useful for consistent feedback and experimentation, but it is not a clinical outcome measure until validated against an appropriate clinical protocol and reference measurements.",
        ],
        styles,
        bullets=[
            "Use a stable camera and keep the full body visible.",
            "Treat missing or uncertain landmarks as invalid rather than assuming a correct measurement.",
            "Validate exercise thresholds with physiotherapy expertise before clinical use.",
        ],
    ))

    story.extend([
        Spacer(1, 0.15 * inch),
        Paragraph("Generated from the current Physiotherapy AI codebase.", styles["SmallGuide"]),
    ])

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return output_path


if __name__ == "__main__":
    path = create_guide()
    print(f"Created PDF: {path}")
