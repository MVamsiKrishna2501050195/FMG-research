from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree


DOC = Path('manuscript/TCSST_IRO_FMG_revision_R11_final.docx')
TMP = DOC.with_suffix('.r18.tmp.docx')
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W_NS}
BOUNDARY = 'supplementary reproducibility files.All-model extension.'


def text_of(p):
    return ''.join(p.xpath('.//w:t/text()', namespaces=NS))


def make_paragraph(ppr, rpr, text):
    p = etree.Element(f'{{{W_NS}}}p', nsmap=ppr.nsmap if ppr is not None else None)
    if ppr is not None:
        p.append(deepcopy(ppr))
    r = etree.SubElement(p, f'{{{W_NS}}}r')
    if rpr is not None:
        r.append(deepcopy(rpr))
    t = etree.SubElement(r, f'{{{W_NS}}}t')
    if text[:1].isspace() or text[-1:].isspace():
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = text
    return p


with ZipFile(DOC, 'r') as zin:
    root = etree.fromstring(zin.read('word/document.xml'))
    matches = []
    for p in root.xpath('.//w:p', namespaces=NS):
        t = text_of(p)
        if BOUNDARY in t:
            matches.append((p, t))
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one joined paragraph, found {len(matches)}')

    p, full = matches[0]
    left, right = full.split(BOUNDARY, 1)
    left += 'supplementary reproducibility files.'
    right = 'All-model extension.' + right
    ppr = p.find(f'{{{W_NS}}}pPr')
    first_run = p.find(f'{{{W_NS}}}r')
    rpr = first_run.find(f'{{{W_NS}}}rPr') if first_run is not None else None
    p1 = make_paragraph(ppr, rpr, left)
    p2 = make_paragraph(ppr, rpr, right)
    parent = p.getparent()
    pos = parent.index(p)
    parent.remove(p)
    parent.insert(pos, p2)
    parent.insert(pos, p1)

    new_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    with ZipFile(TMP, 'w', ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = new_xml if item.filename == 'word/document.xml' else zin.read(item.filename)
            zout.writestr(item, data)

TMP.replace(DOC)
print('Restored paragraph break before All-model extension.')
