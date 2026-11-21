from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
from lxml import etree
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt
from docx.oxml import OxmlElement


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / 'manuscript' / 'TCSST_IRO_FMG_revision_R11_final.docx'
SUPP = ROOT / 'manuscript' / 'FMG_supplementary_public_health_extension_R1.docx'
OUT = ROOT / 'analysis' / 'outputs' / 'review_additions'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W_NS}


def ptext(p):
    return ''.join(p.xpath('.//w:t/text()', namespaces=NS))


def yellow_run(p, text):
    r = etree.SubElement(p, f'{{{W_NS}}}r')
    rpr = etree.SubElement(r, f'{{{W_NS}}}rPr')
    hi = etree.SubElement(rpr, f'{{{W_NS}}}highlight')
    hi.set(f'{{{W_NS}}}val', 'yellow')
    t = etree.SubElement(r, f'{{{W_NS}}}t')
    t.text = text


def rebuild_yellow(p, text):
    ppr = p.find(f'{{{W_NS}}}pPr')
    for child in list(p):
        if child.tag != f'{{{W_NS}}}pPr':
            p.remove(child)
    yellow_run(p, text)


def update_main():
    tmp = MAIN.with_suffix('.r20.tmp.docx')
    with ZipFile(MAIN, 'r') as zin:
        root = etree.fromstring(zin.read('word/document.xml'))
        paragraphs = root.xpath('.//w:p', namespaces=NS)

        # Add the substantive conclusion for the corrected resampled tests.
        target = next(p for p in paragraphs if 'full results are in Supplementary Table S10.' in ptext(p))
        if 'Holm-adjusted p-values were below 0.05' not in ptext(target):
            yellow_run(target, ' Holm-adjusted p-values were below 0.05 in all four Adult and all four Taiwan cells, whereas none of the four COMPAS cells were below 0.05 (Supplementary Table S10).')

        # Temper the descriptive 0.01 statement on the primary-results page.
        target = next(p for p in paragraphs if 'whereas on COMPAS only the MLP consistently exceeds 0.01' in ptext(p))
        old = 'whereas on COMPAS only the MLP consistently exceeds 0.01 (mean FMG 0.0293; 10/10 positive splits).'
        new = 'whereas COMPAS means range from -0.0060 to 0.0293, with the MLP the only model exceeding 0.01 (10/10 positive splits).'
        rebuild_yellow(target, ptext(target).replace(old, new))

        # Correct the S11 label and add the genuine race-group sensitivity.
        target = next(p for p in paragraphs if 'As an alternative focus-group definition' in ptext(p))
        text = ptext(target)
        old = 'As an alternative focus-group definition, we recomputed FMG with g1 treated as the focus group under the same archived masks; mean g1 FMG ranged from 0.0928 to 0.1189 on Adult Income, -0.0349 to -0.0113 on COMPAS, and 0.0742 to 0.0856 on Taiwan Credit (Supplementary Table S11).'
        new = 'As a complement-group sensitivity, we recomputed FMG with g1 treated as the focus group under the same archived masks; mean g1 FMG ranged from 0.0928 to 0.1189 on Adult Income, -0.0349 to -0.0113 on COMPAS, and 0.0742 to 0.0856 on Taiwan Credit (Supplementary Table S11). A separate Adult race-group sensitivity treated non-White records as the focus group and White records as the comparison group; mean FMG ranged from 0.0679 to 0.1106 across the four models, with 10/10 positive splits for every model (Supplementary Table S13).'
        rebuild_yellow(target, text.replace(old, new))

        new_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        with ZipFile(tmp, 'w', ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = new_xml if item.filename == 'word/document.xml' else zin.read(item.filename)
                zout.writestr(item, data)
    tmp.replace(MAIN)


def shade(run, size=8):
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    run.font.size = Pt(size)


def table_with_header(doc, frame, columns):
    table = doc.add_table(rows=1, cols=len(columns))
    table.style = 'Table Grid'
    trpr = table.rows[0]._tr.get_or_add_trPr()
    trpr.append(OxmlElement('w:tblHeader'))
    for j, col in enumerate(columns):
        r = table.rows[0].cells[j].paragraphs[0].add_run(col)
        r.bold = True; shade(r)
    for _, row in frame.iterrows():
        cells = table.add_row().cells
        for j, col in enumerate(columns):
            val = row[col]
            if isinstance(val, float): val = f'{val:.4f}'
            r = cells[j].paragraphs[0].add_run(str(val)); shade(r)
    return table


def update_supplement():
    doc = Document(SUPP)
    for p in doc.paragraphs:
        if p.text.startswith('Supplementary Table S11.'):
            p.text = p.text.replace('Alternative focus-group sensitivity', 'Complement-group sensitivity')
            for r in p.runs: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            break
    data = pd.read_csv(OUT / 'adult_race_group_summary.csv')
    p = doc.add_paragraph(); r = p.add_run('Supplementary Table S13. Adult Income race-group sensitivity at 20% MCAR.'); r.bold = True; shade(r)
    table_with_header(doc, data, ['dataset', 'model', 'mean_fmg', 'sd_fmg', 'min_fmg', 'max_fmg', 'positive_splits', 'mean_focus_positive_n'])
    p = doc.add_paragraph(); r = p.add_run('The focus group is non-White and the comparison group is White. The analysis reuses the archived row-level predictions, verifies labels against reconstructed test splits, and does not retrain the models. Results are descriptive sensitivity evidence.'); shade(r)
    doc.save(SUPP)


if __name__ == '__main__':
    update_main(); update_supplement(); print('R20 manuscript and supplement updates applied.')
