from pathlib import Path
from copy import deepcopy
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import RGBColor, Inches

DOCS = [Path('manuscript/TCSST_IRO_FMG_revision_R11_final.docx'), Path('manuscript/FMG_supplementary_public_health_extension_R1.docx')]

def format_tables(doc):
    for table in doc.tables:
        tbl_pr = table._tbl.tblPr
        borders = tbl_pr.find(qn('w:tblBorders'))
        if borders is None:
            borders = OxmlElement('w:tblBorders'); tbl_pr.append(borders)
        for edge in ('top','left','bottom','right','insideH','insideV'):
            node = borders.find(qn('w:'+edge))
            if node is None:
                node = OxmlElement('w:'+edge); borders.append(node)
            node.set(qn('w:val'),'single'); node.set(qn('w:sz'),'4'); node.set(qn('w:space'),'0'); node.set(qn('w:color'),'000000')
        for row in table.rows:
            for cell in row.cells:
                tc_pr = cell._tc.get_or_add_tcPr(); shd = tc_pr.find(qn('w:shd'))
                if shd is None: shd = OxmlElement('w:shd'); tc_pr.append(shd)
                shd.set(qn('w:fill'),'FFFFFF'); shd.set(qn('w:val'),'clear')
                for p in cell.paragraphs:
                    for run in p.runs: run.font.color.rgb = RGBColor(0,0,0)

def format_equations(doc):
    numbers = ('(2a)','(2b)','(3)','(4)','(5)','(6)','(7)','(8)','(10)','(14)','(15)','(16)')
    changed=0
    for p in doc.paragraphs:
        text=p.text.strip(); number=next((n for n in numbers if text.startswith(n+' ')),None)
        if not number: continue
        formula=text[len(number):].strip(); first=p.runs[0] if p.runs else None
        rpr=deepcopy(first._r.get_or_add_rPr()) if first is not None else None
        for run in list(p.runs): p._p.remove(run._r)
        fr=p.add_run(formula)
        if rpr is not None: fr._r.get_or_add_rPr().extend(rpr)
        p.add_run('\t'+number); p.alignment=WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.tab_stops.add_tab_stop(Inches(6.3), WD_TAB_ALIGNMENT.RIGHT); changed+=1
    return changed

for path in DOCS:
    doc=Document(path); format_tables(doc); count=format_equations(doc); doc.save(path); print(path, count)
