from pathlib import Path
from docx import Document

ROOT = Path(__file__).resolve().parent
DOC = ROOT / "manuscript" / "TCSST_IRO_FMG_revision_R11_final.docx"

LEAD_INS = {
    "The related-work comparison is summarized in Table 1.",
    "Dataset sizes, targets, and focus groups are listed in Table 2.",
    "The overall audit workflow is shown in Figure 1.",
    "The clean and masked data paths are detailed in Figure 2.",
    "The primary benchmark estimates are reported in Table 3.",
    "Masking-rate trajectories are shown in Figure 3.",
    "The component-loss comparison is shown in Figure 4.",
    "The group-conditional sensitivity is summarized in Figure 5.",
    "The Adult Income sensitivity and base-rate analysis are shown in Figure 6.",
    "The extended COMPAS split distributions are shown in Figure 7.",
    "The exploratory attribution changes are shown in Figure 8.",
    "The type-aware imputation results are reported in Table 4.",
    "The type-aware model sensitivity is shown in Figure 9.",
    "The FDA comparison, including the fit-time ratio, is summarized in Table 5.",
    "The RF ReMasker comparison is summarized in Table 6.",
    "The aggregate ReMasker extension is reported in Supplementary Table S9.",
}


def remove_leadins():
    doc = Document(DOC)
    removed = []
    for p in list(doc.paragraphs):
        if p.text.strip() in LEAD_INS:
            removed.append(p.text.strip())
            p._element.getparent().remove(p._element)
    doc.save(DOC)
    print("Removed", len(removed), "standalone visual lead-ins")
    for item in removed:
        print(item)


if __name__ == "__main__":
    remove_leadins()
