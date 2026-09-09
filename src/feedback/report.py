from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_report(session_summary):
    """Create a PDF summary of the session."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Physiotherapy Session Summary")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(60, 760, "Physiotherapy Session Summary")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(60, 730, f"Overall score: {session_summary.overall_score}")

    y = 700
    for result in session_summary.exercise_results:
        pdf.drawString(60, y, f"- {result.exercise}: {result.status} ({result.score})")
        y -= 20
        for suggestion in result.suggestions[:2]:
            pdf.drawString(80, y, f"  * {suggestion}")
            y -= 16

    pdf.save()
    return buffer.getvalue()
