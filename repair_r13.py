from pathlib import Path
from docx import Document
from docx.enum.text import WD_COLOR_INDEX

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "manuscript" / "TCSST_IRO_FMG_revision_R11_final.docx"


def highlight(p):
    for r in p.runs:
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def repair_main():
    doc = Document(MAIN)
    for p in doc.paragraphs:
        if "raw-feature type-aware runner" in p.text and "+0.0169" in p.text:
            text = p.text.replace("(8/10 positive seeds)", "(10/10 positive seeds)")
            old = "These outputs are retained as separate sensitivity analyses rather than pooled as one estimand."
            new = ("These outputs are retained as separate sensitivity analyses rather than pooled as one estimand. "
                   "The discrepancy is expected because the cross-mechanism runner and the raw-feature type-aware runner "
                   "use different masking draws and preprocessing/feature representations; they are therefore not replicate "
                   "estimates of a single fixed pipeline.")
            text = text.replace(old, new)
            p.text = text
            highlight(p)
        if "fmg-submission-release-v5-2026-11-20" in p.text:
            p.text = p.text.replace("fmg-submission-release-v5-2026-11-20", "fmg-submission-release-v6-2026-11-20")
            highlight(p)
    doc.save(MAIN)


if __name__ == "__main__":
    repair_main()
    print("R13 main repair complete")
