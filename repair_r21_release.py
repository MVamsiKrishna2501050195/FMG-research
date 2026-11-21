from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
from docx import Document
from docx.enum.text import WD_COLOR_INDEX


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / 'manuscript' / 'TCSST_IRO_FMG_revision_R11_final.docx'
SUPP = ROOT / 'manuscript' / 'FMG_supplementary_public_health_extension_R1.docx'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W_NS}


def ptext(p):
    return ''.join(p.xpath('.//w:t/text()', namespaces=NS))


def rebuild_yellow(p, text):
    ppr = p.find(f'{{{W_NS}}}pPr')
    for child in list(p):
        if child.tag != f'{{{W_NS}}}pPr': p.remove(child)
    r = etree.SubElement(p, f'{{{W_NS}}}r')
    rpr = etree.SubElement(r, f'{{{W_NS}}}rPr')
    hi = etree.SubElement(rpr, f'{{{W_NS}}}highlight'); hi.set(f'{{{W_NS}}}val', 'yellow')
    t = etree.SubElement(r, f'{{{W_NS}}}t'); t.text = text


def update_main():
    tmp = MAIN.with_suffix('.r21.tmp.docx')
    with ZipFile(MAIN, 'r') as zin:
        root = etree.fromstring(zin.read('word/document.xml'))
        paragraphs = root.xpath('.//w:p', namespaces=NS)
        # The tested package is v14, which contains S10, S11, S12, and S13.
        xml = etree.tostring(root)
        xml = xml.replace(b'fmg-submission-release-v6-2026-11-20', b'fmg-submission-release-v15-2026-11-21')
        xml = xml.replace(b'fmg-submission-release-v14-2026-11-21', b'fmg-submission-release-v15-2026-11-21')
        root = etree.fromstring(xml)
        paragraphs = root.xpath('.//w:p', namespaces=NS)
        p = next(p for p in paragraphs if 'On COMPAS Random Forest' in ptext(p) and 'Supplementary Table S12' in ptext(p))
        rebuild_yellow(p, ptext(p).replace('Supplementary Table S12', 'Supplementary Table S11'))
        p = next(p for p in paragraphs if 'As a complement-group sensitivity' in ptext(p) and 'Supplementary Table S11' in ptext(p))
        rebuild_yellow(p, ptext(p).replace('Supplementary Table S11', 'Supplementary Table S12'))
        new_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        with ZipFile(tmp, 'w', ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = new_xml if item.filename == 'word/document.xml' else zin.read(item.filename)
                zout.writestr(item, data)
    tmp.replace(MAIN)


def paragraph_text(p):
    return p.text


def update_supplement():
    doc = Document(SUPP)
    body = doc._element.body
    children = list(body)
    captions = {}
    for p in doc.paragraphs:
        if p.text.startswith('Supplementary Table S11.'):
            captions['old_s11'] = p
        elif p.text.startswith('Supplementary Table S12.'):
            captions['old_s12'] = p
        elif p.text.startswith('Supplementary Table S13.'):
            captions['s13'] = p
    p11, p12, p13 = captions['old_s11']._p, captions['old_s12']._p, captions['s13']._p
    i11, i12, i13 = children.index(p11), children.index(p12), children.index(p13)
    # Each new table block runs from its caption to the next supplementary caption.
    block11 = children[i11:i12]
    block12 = children[i12:i13]
    for el in block11 + block12:
        body.remove(el)
    insert_at = list(body).index(p13)
    for el in block12 + block11:
        body.insert(insert_at, el); insert_at += 1

    captions['old_s12'].text = 'Supplementary Table S11. COMPAS Random Forest SHAP sensitivity with 100 matched observations per group.'
    captions['old_s11'].text = 'Supplementary Table S12. Complement-group sensitivity using g1 as the focus group under the same archived masks.'
    for run in captions['old_s12'].runs:
        run.bold = True; run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    for run in captions['old_s11'].runs:
        run.bold = True; run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    doc.save(SUPP)


if __name__ == '__main__':
    update_main(); update_supplement(); print('v14 links and S11/S12 order corrected.')
