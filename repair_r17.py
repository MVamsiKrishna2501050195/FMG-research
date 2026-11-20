from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree

DOC = Path('manuscript/TCSST_IRO_FMG_revision_R11_final.docx')
TMP = DOC.with_suffix('.r17.tmp.docx')
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W_NS}


def paragraph_text(p):
    return ''.join(p.xpath('.//w:t/text()', namespaces=NS))


def mark_yellow(p):
    for run in p.xpath('./w:r', namespaces=NS):
        rpr = run.find(f'{{{W_NS}}}rPr')
        if rpr is None:
            rpr = etree.Element(f'{{{W_NS}}}rPr')
            run.insert(0, rpr)
        hi = rpr.find(f'{{{W_NS}}}highlight')
        if hi is None:
            hi = etree.SubElement(rpr, f'{{{W_NS}}}highlight')
        hi.set(f'{{{W_NS}}}val', 'yellow')


replacements = [
    (
        'The detailed Random Forest comparison is shown in Table 4;',
        'The detailed Random Forest comparison is shown in Table 4 and Fig. 9;',
        'Figure 9',
    ),
    (
        'with Holm adjustment across the 12 cells.',
        'with Holm adjustment across the 12 cells (Table 5).',
        'Table 5',
    ),
    (
        'gave descriptive p-values of 0.002, 0.305, and 0.014',
        'gave descriptive p-values of 0.002, 0.305, and 0.014 (Table 6)',
        'Table 6',
    ),
    (
        'after Holm adjustment.',
        'after Holm adjustment (Supplementary Table S9).',
        'Supplementary Table S9',
    ),
]


with ZipFile(DOC, 'r') as zin:
    xml = etree.fromstring(zin.read('word/document.xml'))
    changed = []
    for old, new, label in replacements:
        found = None
        for p in xml.xpath('.//w:p', namespaces=NS):
            text = paragraph_text(p)
            if old in text:
                found = p
                break
        if found is None:
            raise RuntimeError(f'Could not find target for {label}: {old!r}')
        text = paragraph_text(found)
        updated = text.replace(old, new, 1)
        # Rebuild only the textual runs of the target paragraph, preserving pPr.
        for child in list(found):
            if child.tag != f'{{{W_NS}}}pPr':
                found.remove(child)
        run = etree.SubElement(found, f'{{{W_NS}}}r')
        rpr = etree.SubElement(run, f'{{{W_NS}}}rPr')
        text_node = etree.SubElement(run, f'{{{W_NS}}}t')
        if updated[:1].isspace() or updated[-1:].isspace():
            text_node.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        text_node.text = updated
        mark_yellow(found)
        changed.append(label)

    new_xml = etree.tostring(xml, xml_declaration=True, encoding='UTF-8', standalone=True)
    with ZipFile(TMP, 'w', ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = new_xml if item.filename == 'word/document.xml' else zin.read(item.filename)
            zout.writestr(item, data)

TMP.replace(DOC)
print('Updated:', ', '.join(changed))
