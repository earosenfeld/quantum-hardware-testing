import pandas as pd
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_csv_report(log_entries, filename='data/processed/test_log.csv'):
    df = pd.DataFrame(log_entries)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    return filename

def generate_pdf_report(log_entries, filename='reports/test_report.pdf'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.drawString(100, height - 40, "Cryocooler Test Report")
    y = height - 60
    for entry in log_entries[:40]:  # limit to 40 lines for demo
        c.drawString(50, y, str(entry))
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 40
    c.save()
    return filename 