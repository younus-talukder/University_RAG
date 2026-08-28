from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT_DIR / "docs" / "PROJECT_PROCESS.md"
OUTPUT_PATH = ROOT_DIR / "docs" / "University_RAG_Project_Process.pdf"


def parse_markdown(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            blocks.append(("code", "\n".join(code)))
            index += 1
            continue
        if line.startswith("# "):
            blocks.append(("title", line[2:]))
        elif line.startswith("## "):
            blocks.append(("h2", line[3:]))
        elif line.startswith("### "):
            blocks.append(("h3", line[4:]))
        elif line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(lines[index][2:])
                index += 1
            blocks.append(("bullets", items))
            continue
        elif line[:2].isdigit() and line[2:4] == ". ":
            items = []
            while index < len(lines) and lines[index][:2].isdigit() and lines[index][2:4] == ". ":
                items.append(lines[index][4:])
                index += 1
            blocks.append(("numbered", items))
            continue
        else:
            paragraph = line
            index += 1
            while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "- ", "```")):
                paragraph += " " + lines[index].strip()
                index += 1
            blocks.append(("paragraph", paragraph))
            continue
        index += 1
    return blocks


def inline_markup(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("`", "")


class SectionRule(Flowable):
    def __init__(self):
        super().__init__()
        self.height = 3 * mm

    def draw(self):
        self.canv.setStrokeColor(colors.HexColor("#19A7A0"))
        self.canv.setLineWidth(1.2)
        self.canv.line(0, 1.5 * mm, 32 * mm, 1.5 * mm)


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7DEE5"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 9 * mm, "University Information Chatbot | Project Process Documentation")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_pdf():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=27, leading=32, alignment=TA_CENTER, textColor=colors.HexColor("#123047"),
        spaceAfter=8 * mm,
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=13, leading=19, alignment=TA_CENTER, textColor=colors.HexColor("#536575"),
    ))
    styles.add(ParagraphStyle(
        name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=16, leading=20, textColor=colors.HexColor("#123047"), spaceBefore=7 * mm, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="H3Custom", parent=styles["Heading3"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=15, textColor=colors.HexColor("#087F7A"), spaceBefore=4 * mm, spaceAfter=1.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.5, leading=14, textColor=colors.HexColor("#263746"), spaceAfter=2.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="BulletCustom", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.2, leading=13, leftIndent=7 * mm, firstLineIndent=-4 * mm,
        bulletIndent=2 * mm, textColor=colors.HexColor("#263746"), spaceAfter=1.3 * mm,
    ))
    styles.add(ParagraphStyle(
        name="CodeCustom", parent=styles["Code"], fontName="Courier",
        fontSize=8.2, leading=11, leftIndent=5 * mm, rightIndent=5 * mm,
        backColor=colors.HexColor("#F1F5F7"), borderColor=colors.HexColor("#D7DEE5"),
        borderWidth=0.5, borderPadding=5 * mm, spaceBefore=2 * mm, spaceAfter=4 * mm,
    ))

    document = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=19 * mm, title="University RAG Project Process",
        author="University Information Chatbot Project",
    )
    story = []
    story.extend([
        Spacer(1, 32 * mm),
        Paragraph("University Information Chatbot", styles["CoverTitle"]),
        Paragraph("A-to-Z Project Process and Technical Documentation", styles["CoverSubtitle"]),
        Spacer(1, 15 * mm),
        SectionRule(),
        Spacer(1, 8 * mm),
        Paragraph("A complete description of how the PDF-grounded multilingual RAG system is built, executed, evaluated, and troubleshot.", styles["CoverSubtitle"]),
        Spacer(1, 35 * mm),
    ])
    summary_data = [
        [Paragraph("Knowledge base", styles["BodyCustom"]), Paragraph("Official university PDF documents", styles["BodyCustom"])],
        [Paragraph("Languages", styles["BodyCustom"]), Paragraph("English, Bangla, Banglish", styles["BodyCustom"])],
        [Paragraph("Default UI mode", styles["BodyCustom"]), Paragraph("Fast extractive answering", styles["BodyCustom"])],
        [Paragraph("Optional mode", styles["BodyCustom"]), Paragraph("Local Qwen generation", styles["BodyCustom"])],
        [Paragraph("Vector store", styles["BodyCustom"]), Paragraph("FAISS IndexFlatIP + metadata.pkl", styles["BodyCustom"])],
    ]
    table = Table(summary_data, colWidths=[45 * mm, 105 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E5F4F2")),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#F7FAFB")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#B9D8D5")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D7DEE5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(PageBreak())

    for kind, value in parse_markdown(SOURCE_PATH):
        if kind == "title":
            continue
        if kind == "h2":
            story.extend([Paragraph(inline_markup(value), styles["H2Custom"]), SectionRule()])
        elif kind == "h3":
            story.append(Paragraph(inline_markup(value), styles["H3Custom"]))
        elif kind == "paragraph":
            story.append(Paragraph(inline_markup(value), styles["BodyCustom"]))
        elif kind == "code":
            story.append(Preformatted(value, styles["CodeCustom"]))
        elif kind in {"bullets", "numbered"}:
            for item_number, item in enumerate(value, start=1):
                marker = "-" if kind == "bullets" else f"{item_number}."
                story.append(Paragraph(f"{marker} {inline_markup(item)}", styles["BulletCustom"]))

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
