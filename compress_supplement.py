from docx import Document
from docx.shared import Pt

p='manuscript/FMG_supplementary_public_health_extension_R1.docx'
d=Document(p)
for para in d.paragraphs:
    para.paragraph_format.space_before=Pt(0); para.paragraph_format.space_after=Pt(0); para.paragraph_format.line_spacing=1
    if para.style.name=='Caption':
        for r in para.runs: r.font.size=Pt(10)
for table in d.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                para.paragraph_format.space_before=Pt(0); para.paragraph_format.space_after=Pt(0); para.paragraph_format.line_spacing=1
                for r in para.runs: r.font.size=Pt(8)
d.save(p); print('compressed supplementary tables')
