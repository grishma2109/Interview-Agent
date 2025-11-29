from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pathlib import Path

out = Path("data/policies.pdf")
out.parent.mkdir(exist_ok=True)

c = canvas.Canvas(str(out), pagesize=A4)
width, height = A4

c.setFont("Helvetica-Bold", 16)
c.drawString(60, height - 80, "ACME Corp — Employee Policies")
c.setFont("Helvetica", 11)
text = c.beginText(60, height - 120)
paragraphs = [
    "1. Leave Policy:",
    "Employees are entitled to 12 paid leaves per year. Casual leave must be applied 24 hours in advance when possible.",
    "",
    "2. Maternity/Paternity Leave:",
    "Maternity leave is 26 weeks for eligible employees. Paternity leave is 15 days.",
    "",
    "3. Benefits:",
    "Health insurance is provided from the first day of employment. Employees may enroll dependents during the annual enrollment period.",
    "",
    "4. Notice Period:",
    "For resignation, employees must provide 30 days notice or pay in lieu of notice as per company policy.",
    "",
    "If you need help with any policy, contact HR at hr@acme.example.com."
]

for p in paragraphs:
    text.textLine(p)
c.drawText(text)
c.showPage()
c.save()
print(f"Sample PDF created at: {out}")
