from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import textwrap
import os
import requests


pdfmetrics.registerFont(TTFont("CourierNew", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))


def draw_border(c, width, height):
    c.setLineWidth(2)
    c.rect(0.5*inch, 0.5*inch, width - inch, height - inch)



from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics



def generate_pdf(report_text, radiology_images, output_file, logo_path=None):
    doc = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter

    # ---- PAGE 1: Hospital Logo + Title ----
    draw_border(doc, width, height)

    if logo_path and os.path.exists(logo_path):
        doc.drawImage(logo_path, 0.75*inch, height - 2*inch, width=2*inch, preserveAspectRatio=True)

    doc.setFont("CourierNew", 14)
    doc.drawString(0.75*inch, height - 2.5*inch, "SYNTHETIC MEDICAL RECORD")

    doc.setFont("CourierNew", 10)
    y = height - 3*inch

    # ---- MAIN TEXT ----
    lines = report_text.split("\n")

    for line in lines:
        if y < inch:
            draw_border(doc, width, height)
            doc.showPage()
            doc.setFont("CourierNew", 10)
            y = height - inch

        doc.drawString(0.75*inch, y, line)
        y -= 12

    # ---- RADIOLOGY IMAGE PAGES ----
    for img_url in radiology_images:
        draw_border(doc, width, height)
        doc.showPage()
        img_data = requests.get(img_url).content
        temp_path = "temp_image.png"
        with open(temp_path, "wb") as f:
            f.write(img_data)

        doc.drawImage(temp_path, 0.75*inch, 1.5*inch, width=7*inch, preserveAspectRatio=True)
        doc.setFont("CourierNew", 12)
        doc.drawString(0.75*inch, 0.75*inch, "Radiology Image")

    draw_border(doc, width, height)
    doc.save()
