from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

P='manuscript/FMG_supplementary_public_health_extension_R1.docx'
d=Document(P)
for p in d.paragraphs:
    s=' '.join(p.text.split())
    if s.startswith('All seed-level summaries below'):
        p.text='All seed-level summaries below are descriptive repetitions of train/test splits on the same finite benchmark datasets. The splits are not treated as independent samples. Except where stated in Tables S9 to S11, no p-values or intervals are used in this supplement. Results are limited to the datasets, models, groups, preprocessing, and missingness mechanisms listed in each table.'
    if s.startswith('Data sources:'):
        p.text='Data sources: CDC 2024 BRFSS Survey Data and Documentation [20]; Diabetes 130-US Hospitals for Years 1999–2008 [21]. The extension remains a retrospective benchmark and does not show improved clinical decisions, monitoring benefit, or deployment safety.'
    if s.startswith('Table S') or s.startswith('Supplementary Table S'):
        p.text=s.replace('Table S','Supplementary Table S',1) if s.startswith('Table S') else s
        for r in p.runs: r.bold=True; r.font.name='Times New Roman'; r.font.size=Pt(10)
        p.alignment=WD_ALIGN_PARAGRAPH.LEFT

def header(table, values):
    for cell,value in zip(table.rows[0].cells,values):
        cell.text=value
        for p in cell.paragraphs:
            for r in p.runs: r.bold=True; r.font.size=Pt(8)

header(d.tables[7], ['Dataset','ΔTPR (all)','ΔTNR (all)','ΔA','FMG (all)'])
header(d.tables[9], ['Dataset','Model','Mean FMG','Corrected SE','t statistic','Holm-adjusted p','95% CI low','95% CI high','Positive splits'])
header(d.tables[10], ['Focus group','Feature','Matched observations','Mean |SHAP| clean','Mean |SHAP| masked','Mean |SHAP| shift','95% CI low','95% CI high'])
header(d.tables[11], ['Dataset','Model','Prespecified g0 mean FMG','Complement g1 mean FMG','SD','Minimum','Maximum','Positive splits'])
header(d.tables[12], ['Dataset','Model','Mean FMG','SD','Minimum','Maximum','Positive splits','Mean focus-group positive count'])
for row in d.tables[12].rows[1:]:
    for c in row.cells:
        if c.text.strip()=='202.7000': c.text='202.7'
for table in d.tables:
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1
                for r in p.runs: r.font.name='Times New Roman'; r.font.size=Pt(8)

def subscript_header(cell, base, index):
    cell.text=''
    p=cell.paragraphs[0]
    r=p.add_run(base); r.font.size=Pt(8); r.bold=True
    r=p.add_run(index); r.font.size=Pt(8); r.bold=True; r.font.subscript=True

subscript_header(d.tables[7].cell(0,1), 'ΔTPR', 'all')
subscript_header(d.tables[7].cell(0,2), 'ΔTNR', 'all')
subscript_header(d.tables[7].cell(0,4), 'FMG', 'all')
d.save(P); print('Supplement corrected')
