from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement


path = Path('manuscript/FMG_supplementary_public_health_extension_R1.docx')
doc = Document(path)
for table in doc.tables[-3:]:
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    if not tr_pr.xpath('./w:tblHeader'):
        tr_pr.append(OxmlElement('w:tblHeader'))
doc.save(path)
print('Marked the three new supplementary table header rows to repeat across pages.')
