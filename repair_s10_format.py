from docx import Document
from docx.enum.text import WD_COLOR_INDEX


path = 'manuscript/FMG_supplementary_public_health_extension_R1.docx'
doc = Document(path)
changed = 0
for table in doc.tables:
    headers = [c.text.strip() for c in table.rows[0].cells]
    if 'p_holm' not in headers:
        continue
    col = headers.index('p_holm')
    for row in table.rows[1:]:
        cell = row.cells[col]
        if cell.text.strip() == '0.0000':
            cell.text = '<0.0001'
            for run in cell.paragraphs[0].runs:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
            changed += 1
doc.save(path)
print('formatted Holm p-values:', changed)
