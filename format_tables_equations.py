from pathlib import Path
from copy import deepcopy
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import RGBColor, Inches, Pt

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

def format_template(doc):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.top_margin, sec.bottom_margin = Inches(1.04), Inches(0.66)
    sec.left_margin, sec.right_margin = Inches(1), Inches(1)
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'; normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    normal.paragraph_format.line_spacing = 1
    for name in ('Heading 1','Heading 2','Heading 3','Caption'):
        if name in doc.styles:
            st=doc.styles[name]; st.font.name='Times New Roman'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'Times New Roman')
    for i,p in enumerate(doc.paragraphs):
        size = 18 if i == 0 else 12
        bold = True if i in (0,1) or p.style.name.startswith('Heading') or p.style.name == 'Caption' else None
        for r in p.runs:
            r.font.name='Times New Roman'; r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Times New Roman'); r.font.size=Pt(size); r.font.color.rgb=RGBColor(0,0,0)
            if bold is not None: r.bold=bold
        p.paragraph_format.line_spacing=1
        p.paragraph_format.space_after=Pt(0)
        p.paragraph_format.space_before=Pt(0)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing=1
                    p.paragraph_format.space_after=Pt(0)
                    p.paragraph_format.space_before=Pt(0)
                    for r in p.runs: r.font.name='Times New Roman'; r.font.size=Pt(10)

def convert_equations_to_omml(doc):
    numbers=('(2a)','(2b)','(3)','(4)','(5)','(6)','(7)','(8)','(10)','(14)','(15)','(16)')
    changed=0
    for p in doc.paragraphs:
        text=p.text.strip()
        number=next((n for n in numbers if text.endswith('\t'+n)),None)
        if not number: continue
        formula=text[:-len(number)].rstrip()
        for run in list(p.runs): p._p.remove(run._r)
        omath=OxmlElement('m:oMath'); mr=OxmlElement('m:r'); mt=OxmlElement('m:t'); mt.text=formula; mr.append(mt); omath.append(mr); p._p.append(omath)
        p.add_run('\t'+number); p.alignment=WD_ALIGN_PARAGRAPH.LEFT; p.paragraph_format.tab_stops.add_tab_stop(Inches(6.3), WD_TAB_ALIGNMENT.RIGHT); changed+=1
    return changed

for path in DOCS:
    doc=Document(path); format_tables(doc); format_template(doc); count=format_equations(doc); omml=convert_equations_to_omml(doc); doc.save(path); print(path, count, 'OMML', omml)
