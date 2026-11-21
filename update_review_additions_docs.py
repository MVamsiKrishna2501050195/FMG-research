from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
from lxml import etree
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / 'manuscript' / 'TCSST_IRO_FMG_revision_R11_final.docx'
SUPP = ROOT / 'manuscript' / 'FMG_supplementary_public_health_extension_R1.docx'
OUT = ROOT / 'analysis' / 'outputs' / 'review_additions'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W_NS}


def ptext(p):
    return ''.join(p.xpath('.//w:t/text()', namespaces=NS))


def append_highlighted_run(p, text):
    r = etree.SubElement(p, f'{{{W_NS}}}r')
    rpr = etree.SubElement(r, f'{{{W_NS}}}rPr')
    hi = etree.SubElement(rpr, f'{{{W_NS}}}highlight')
    hi.set(f'{{{W_NS}}}val', 'yellow')
    t = etree.SubElement(r, f'{{{W_NS}}}t')
    t.text = text


def update_main():
    replacements = [
        (
            'The primary unit of comparison is the dataset-model cell; the ten seed-level values are repeated split realizations within each cell, not independent population samples.',
            'The primary unit of comparison is the dataset-model cell; the ten seed-level values are repeated split realizations within each cell, not independent population samples.',
            ' We additionally report a Nadeau--Bengio corrected resampled t-test of mean FMG against zero for the 12 primary cells, using the 70:30 test-to-training ratio; full results are in Supplementary Table S10.',
        ),
        (
            'Across the 12 primary pairs, the within-pair Spearman correlation between clean baseline focus-group recall and 20% MCAR FMG ranges widely from -0.394 to +0.806, indicating that baseline recall level alone does not determine FMG magnitude.',
            'Across the 12 primary pairs, the within-pair Spearman correlation between clean baseline focus-group recall and 20% MCAR FMG ranges widely from -0.394 to +0.806, indicating that baseline recall level alone does not determine FMG magnitude.',
            ' As an alternative focus-group definition, we recomputed FMG with g1 treated as the focus group under the same archived masks; mean g1 FMG ranged from 0.0928 to 0.1189 on Adult Income, -0.0349 to -0.0113 on COMPAS, and 0.0742 to 0.0856 on Taiwan Credit (Supplementary Table S11).',
        ),
        (
            'On COMPAS Random Forest (seed 42, 20% MCAR; Fig. 8), the largest between-group shifts in mean absolute SHAP values occurred on priors_count, age, and c_days_from_compas.',
            'On COMPAS Random Forest (seed 42, 20% MCAR; Fig. 8), the largest between-group shifts in mean absolute SHAP values occurred on priors_count, age, and c_days_from_compas.',
            ' Increasing the matched sample to 100 observations per group preserved priors_count as the largest mean absolute SHAP shift in both groups; this remains descriptive attribution sensitivity, not a causal explanation (Supplementary Table S12).',
        ),
    ]
    tmp = MAIN.with_suffix('.review_additions.tmp.docx')
    with ZipFile(MAIN, 'r') as zin:
        root = etree.fromstring(zin.read('word/document.xml'))
        changed = 0
        for marker, expected, addition in replacements:
            found = None
            for p in root.xpath('.//w:p', namespaces=NS):
                if marker in ptext(p):
                    found = p
                    break
            if found is None:
                raise RuntimeError(f'Main-text target not found: {marker[:80]}')
            if addition.strip() in ptext(found):
                continue
            append_highlighted_run(found, addition)
            changed += 1
        new_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        with ZipFile(tmp, 'w', ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = new_xml if item.filename == 'word/document.xml' else zin.read(item.filename)
                zout.writestr(item, data)
    tmp.replace(MAIN)
    print('Main text additions:', changed)


def shade_run(run):
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    run.font.size = Pt(8)


def add_caption(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return p


def add_note(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    r.font.size = Pt(8)


def add_table(doc, frame, columns):
    table = doc.add_table(rows=1, cols=len(columns))
    table.style = 'Table Grid'
    for j, col in enumerate(columns):
        cell = table.rows[0].cells[j]
        run = cell.paragraphs[0].add_run(col)
        run.bold = True
        shade_run(run)
    for _, row in frame.iterrows():
        cells = table.add_row().cells
        for j, col in enumerate(columns):
            value = row[col]
            if isinstance(value, float):
                value = f'{value:.4f}'
            run = cells[j].paragraphs[0].add_run(str(value))
            shade_run(run)
    return table


def update_supplement():
    doc = Document(SUPP)
    s10 = pd.read_csv(OUT / 'corrected_resampled_t_primary.csv')
    s11 = pd.read_csv(OUT / 'alternative_focus_group_g1_summary.csv')
    s12 = pd.read_csv(OUT / 'compas_rf_shap_100_per_feature.csv')

    add_caption(doc, 'Supplementary Table S10. Nadeau--Bengio corrected resampled tests of mean FMG against zero for the 12 primary cells.')
    add_table(doc, s10, ['dataset', 'model', 'mean_fmg', 'corrected_se', 't_statistic', 'p_holm', 'ci95_low', 'ci95_high', 'positive_splits'])
    add_note(doc, 'The correction uses k = 10 repeated 70:30 splits and the Nadeau--Bengio variance factor 1/k + n_test/n_train. P-values are two-sided and Holm-adjusted across the 12 cells. The test addresses repeated-split dependence, not external population inference.')

    add_caption(doc, 'Supplementary Table S11. Alternative focus-group sensitivity using g1 as the focus group under the same archived masks.')
    add_table(doc, s11, ['dataset', 'model', 'prespecified_g0_mean_fmg', 'alternative_g1_mean_fmg', 'alternative_g1_sd', 'alternative_g1_min', 'alternative_g1_max', 'alternative_g1_positive_splits'])
    add_note(doc, 'This is a group-definition sensitivity analysis using the same fitted models, test masks, and ten seeds. It does not establish that either group is intrinsically more vulnerable.')

    add_caption(doc, 'Supplementary Table S12. COMPAS Random Forest SHAP sensitivity with 100 matched observations per group.')
    add_table(doc, s12, ['subgroup', 'feature', 'n_matched', 'mean_abs_shap_clean', 'mean_abs_shap_masked', 'mean_abs_shap_shift', 'ci95_low', 'ci95_high'])
    add_note(doc, 'SHAP values use the positive-class TreeExplainer output for seed 42 with 20% test-only MCAR and training-fitted median imputation. The 95% intervals are descriptive row-level intervals; the analysis is not causal.')
    doc.save(SUPP)
    print('Supplement tables added:', len(s10), len(s11), len(s12))


if __name__ == '__main__':
    update_main()
    update_supplement()
