from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import os
import requests
import textwrap

# Use built-in ReportLab font — no external font files needed (cross-platform)
FONT_BODY = "Courier"
FONT_TITLE = "Courier-Bold"


def draw_border(c, width, height):
    c.setLineWidth(2)
    c.rect(0.5 * inch, 0.5 * inch, width - inch, height - inch)


def generate_pdf(report_text, radiology_images, output_file, logo_path=None):
    doc = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter

    # ---- PAGE 1: Hospital Logo + Title ----
    draw_border(doc, width, height)

    if logo_path and os.path.exists(logo_path):
        doc.drawImage(logo_path, 0.75 * inch, height - 2 * inch,
                      width=2 * inch, preserveAspectRatio=True)

    doc.setFont(FONT_TITLE, 14)
    doc.drawString(0.75 * inch, height - 2.5 * inch, "SYNTHETIC MEDICAL RECORD")

    doc.setFont(FONT_BODY, 10)
    y = height - 3 * inch

    # ---- MAIN TEXT ----
    for line in report_text.split("\n"):
        if y < inch:
            draw_border(doc, width, height)
            doc.showPage()
            doc.setFont(FONT_BODY, 10)
            y = height - inch

        doc.drawString(0.75 * inch, y, line)
        y -= 12

    # ---- RADIOLOGY IMAGE PAGES ----
    for img_url in radiology_images:
        draw_border(doc, width, height)
        doc.showPage()
        img_data = requests.get(img_url).content
        temp_path = "temp_image.png"
        with open(temp_path, "wb") as f:
            f.write(img_data)

        doc.drawImage(temp_path, 0.75 * inch, 1.5 * inch,
                      width=7 * inch, preserveAspectRatio=True)
        doc.setFont(FONT_BODY, 12)
        doc.drawString(0.75 * inch, 0.75 * inch, "Radiology Image")

    draw_border(doc, width, height)
    doc.save()
