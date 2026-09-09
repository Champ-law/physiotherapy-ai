# src/api/server.py
from flask import Flask, jsonify, request

from src.api.schemas import validate_analysis_request
from src.feedback.text_feedback import generate_feedback
from src.rules_engine.squat_rules import analyze_squat

app = Flask(__name__)


@app.route("/analyze_pose", methods=["POST"])
def analyze_pose():
    payload = request.get_json(silent=True) or {}

    try:
        validated = validate_analysis_request(payload)
    except ValueError as exc:
        return jsonify({
            "status": "invalid_payload",
            "score": 0,
            "warnings": [str(exc)],
            "suggestions": ["Provide an object with an 'exercise' and a valid 'landmarks' dictionary."],
        }), 400

    landmarks = validated["landmarks"]
    side = validated["side"]
    exercise = validated["exercise"]

    if exercise == "squat":
        result = analyze_squat(landmarks, side=side)
    else:
        result = {
            "status": "not_implemented",
            "score": 0,
            "warnings": [f"Exercise '{exercise}' is not implemented yet."],
            "suggestions": ["Use the 'squat' exercise for active evaluation."],
            "metrics": {},
        }

    return jsonify(generate_feedback(result))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
