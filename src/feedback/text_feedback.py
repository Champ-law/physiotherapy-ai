# src/feedback/text_feedback.py

def generate_feedback(result):
    if isinstance(result, dict):
        status = result.get("status", "unknown")
        suggestions = result.get("suggestions", ["Adjust movement and maintain posture."])
        warnings = result.get("warnings", [])
        return {
            "status": status,
            "score": result.get("score", 0),
            "warnings": warnings,
            "suggestions": suggestions,
        }

    status = str(result)
    suggestion = "Maintain posture" if status == "good_depth" else "Adjust movement"
    return {
        "status": status,
        "score": 0,
        "warnings": [],
        "suggestions": [suggestion],
    }
