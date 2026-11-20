from pathlib import Path
from docx import Document
from docx.enum.text import WD_COLOR_INDEX

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "manuscript" / "TCSST_IRO_FMG_revision_R11_final.docx"
SUPP = ROOT / "manuscript" / "FMG_supplementary_public_health_extension_R1.docx"


def highlight_paragraph(p):
    for r in p.runs:
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def set_highlighted_cell(cell, text):
    cell.text = text
    for p in cell.paragraphs:
        for r in p.runs:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def repair_main():
    doc = Document(MAIN)
    for p in doc.paragraphs:
        changed = False
        if "+0.025 to +0.044" in p.text:
            p.text = p.text.replace("+0.025 to +0.044", "+0.026 to +0.044")
            changed = True
        if "Supplementary Table S10" in p.text or "Supplementary Table S11" in p.text:
            p.text = p.text.replace("Supplementary Table S10", "Supplementary Table S8")
            p.text = p.text.replace("Supplementary Table S11", "Supplementary Table S9")
            changed = True
        if changed:
            highlight_paragraph(p)
    doc.save(MAIN)


def repair_supplement():
    doc = Document(SUPP)
    for p in doc.paragraphs:
        if "Supplementary Table S10" in p.text or "Supplementary Table S11" in p.text:
            p.text = p.text.replace("Supplementary Table S10", "Supplementary Table S8")
            p.text = p.text.replace("Supplementary Table S11", "Supplementary Table S9")
            highlight_paragraph(p)
    # S8 is the decomposition table; fill the previously suppressed COMPAS
    # accuracy-loss cell from FMG_all = ΔTPR_all − ΔA.
    for table in doc.tables:
        if table.rows and len(table.columns) == 5 and "ΔTPR_all" in table.rows[0].cells[1].text:
            for row in table.rows[1:]:
                if row.cells[0].text.strip() == "COMPAS":
                    set_highlighted_cell(row.cells[3], "0.0251")
    doc.save(SUPP)


if __name__ == "__main__":
    repair_main()
    repair_supplement()
    print("R14 repairs complete")
