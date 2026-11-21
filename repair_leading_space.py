from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


DOC = Path("manuscript/TCSST_IRO_FMG_revision_R11_final.docx")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W}

with NamedTemporaryFile(suffix=".docx", delete=False, dir=DOC.parent) as tmp:
    temp = Path(tmp.name)

with ZipFile(DOC) as zin, ZipFile(temp, "w", ZIP_DEFLATED) as zout:
    changed = False
    for info in zin.infolist():
        data = zin.read(info.filename)
        if info.filename == "word/document.xml":
            root = etree.fromstring(data)
            for paragraph in root.xpath(".//w:p", namespaces=NS):
                text = "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))
                if "Holm-adjusted p-values were below" not in text:
                    continue
                for node in paragraph.xpath(".//w:t", namespaces=NS):
                    if node.text and node.text.endswith("samples."):
                        node.text += " "
                        node.set(f"{{{XML}}}space", "preserve")
                        changed = True
                    if node.text and node.text.startswith(" Holm-adjusted"):
                        node.set(f"{{{XML}}}space", "preserve")
                        changed = True
            data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
        zout.writestr(info, data)

if not changed:
    raise RuntimeError("Target leading-space run was not found")
temp.replace(DOC)
print(f"Fixed leading-space preservation in {DOC}")
