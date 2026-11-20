from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
from PIL import Image

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DOC = ROOT / "manuscript" / "TCSST_IRO_FMG_revision_R11_final.docx"

# Pixels removed from the top of each source image. These strips contain only
# the embedded overall title; panel labels, legends, axes, and data remain.
CROPS = {
    "fig03_masking_gap_curves.png": 120,
    "fig04_accuracy_vs_recall.png": 115,
    "fig05_group_conditional_vs_mcar.png": 115,
    "fig06_class_imbalance.png": 80,
    "fig07_compas_extended_seeds_clean.png": 50,
    "fig08_shap_shifts.png": 65,
    "fig09_imputation_ablation.png": 80,
}

MEDIA = {
    "fig03_masking_gap_curves.png": "word/media/image14.png",
    "fig04_accuracy_vs_recall.png": "word/media/image15.png",
    "fig05_group_conditional_vs_mcar.png": "word/media/image16.png",
    "fig06_class_imbalance.png": "word/media/image17.png",
    "fig07_compas_extended_seeds_clean.png": "word/media/image18.png",
    "fig08_shap_shifts.png": "word/media/image19.png",
    "fig09_imputation_ablation.png": "word/media/image20.png",
}


def crop_sources():
    cropped = {}
    for name, top in CROPS.items():
        src = FIG / name
        with Image.open(src) as im:
            out = im.crop((0, top, im.width, im.height)).convert("RGB")
            out.save(src, format="PNG", optimize=True)
            buf = BytesIO()
            out.save(buf, format="PNG", optimize=True)
            cropped[name] = buf.getvalue()
            # Keep standalone PDF figure assets aligned with the manuscript.
            out.save(FIG / name.replace(".png", ".pdf"), format="PDF", resolution=300.0)
    return cropped


def replace_docx_media(cropped):
    temp = DOC.with_suffix(".tmp.docx")
    with ZipFile(DOC, "r") as zin, ZipFile(temp, "w", ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            for name, media in MEDIA.items():
                if item.filename == media:
                    data = cropped[name]
                    break
            zout.writestr(item, data)
    temp.replace(DOC)


if __name__ == "__main__":
    replace_docx_media(crop_sources())
    print("Embedded figure titles removed from Figures 3-9")
