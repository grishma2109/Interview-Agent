# report_generator.py
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pathlib import Path
import matplotlib.pyplot as plt

def _make_score_chart(qa_history, out_png_path):
    # one bar per question with its score
    labels = [f"Q{i+1}" for i in range(len(qa_history))]
    scores = [q.get("score", 0) for q in qa_history]

    plt.figure(figsize=(6,3))
    plt.bar(labels, scores)  # do not specify colors per instructions
    plt.ylim(0,5)
    plt.xlabel("Questions")
    plt.ylabel("Score (1-5)")
    plt.title("Candidate Scores")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()

def generate_report_pdf(candidate, skills, qa_history, role):
    """
    Returns PDF bytes for a candidate report including a bar chart.
    """
    # create chart PNG in-memory
    png_buf = Path("tmp_chart.png")
    _make_score_chart(qa_history, str(png_buf))

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 60, f"Interview Report — {candidate.get('name','Candidate')}")
    c.setFont("Helvetica", 11)
    c.drawString(40, height - 80, f"Role applied: {role}")
    c.drawString(40, height - 95, f"Email: {candidate.get('email','')}")
    c.drawString(40, height - 110, f"Phone: {candidate.get('phone','')}")
    c.drawString(40, height - 125, f"Generated: {candidate.get('timestamp','')}")

    # Skills
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 150, "Detected Skills:")
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 165, ", ".join(skills) if skills else "None detected")

    # Chart
    c.drawImage(str(png_buf), 40, height - 420, width=500, preserveAspectRatio=True, mask='auto')

    # QA summary
    y = height - 460
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Q&A Summary:")
    y -= 20
    c.setFont("Helvetica", 9)
    for i, qa in enumerate(qa_history, 1):
        if y < 80:
            c.showPage()
            y = height - 60
        c.drawString(45, y, f"{i}. ({qa.get('type')}) Score: {qa.get('score'):.1f} — {qa.get('question')[:80]}")
        y -= 14
        c.drawString(60, y, f"Answer: { (qa.get('answer')[:200] + '...') if len(qa.get('answer'))>200 else qa.get('answer') }")
        y -= 18
        c.drawString(60, y, f"Remarks: {qa.get('remarks')[:200]}")
        y -= 20

    c.showPage()
    c.save()
    pdf_bytes = packet.getvalue()

    try:
        png_buf.unlink()
    except Exception:
        pass

    return pdf_bytes
