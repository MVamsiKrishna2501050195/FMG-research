from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


DOC = Path("manuscript/TCSST_IRO_FMG_revision_R11_final.docx")
old = b"fmg-submission-release-v15-2026-11-21"
new = b"fmg-submission-release-v16-2026-11-21"

with NamedTemporaryFile(suffix=".docx", delete=False, dir=DOC.parent) as tmp:
    temp = Path(tmp.name)

with ZipFile(DOC) as zin, ZipFile(temp, "w", ZIP_DEFLATED) as zout:
    changed = False
    for info in zin.infolist():
        data = zin.read(info.filename)
        if info.filename == "word/document.xml" and old in data:
            data = data.replace(old, new)
            changed = True
        zout.writestr(info, data)

if not changed:
    raise RuntimeError("The v14 release URL was not found")
temp.replace(DOC)
print(f"Updated {DOC} to the v15 release URL")
