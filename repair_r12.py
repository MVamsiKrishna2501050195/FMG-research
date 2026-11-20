from pathlib import Path
import shutil
import pandas as pd
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "manuscript" / "TCSST_IRO_FMG_revision_R11_final.docx"
SUPPLEMENT = ROOT / "manuscript" / "FMG_supplementary_public_health_extension_R1.docx"
CORR = ROOT / "adult_corrected_rerun"


def highlight_run(run):
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def highlight_paragraph(p):
    for r in p.runs:
        highlight_run(r)


def set_cell(cell, text, highlight=False):
    cell.text = str(text)
    if highlight:
        for p in cell.paragraphs:
            for r in p.runs:
                highlight_run(r)


def remove_paragraph(p):
    p._element.getparent().remove(p._element)


def insert_table_before(doc, before_paragraph, rows, cols, headers, data):
    table = doc.add_table(rows=rows, cols=cols)
    table.style = "Table Grid"
    for j, h in enumerate(headers):
        set_cell(table.cell(0, j), h, highlight=True)
    for i, row in enumerate(data, start=1):
        for j, value in enumerate(row):
            set_cell(table.cell(i, j), value, highlight=True)
    before_paragraph._p.addprevious(table._tbl)
    return table


def repair_supplement():
    doc = Document(SUPPLEMENT)

    # The R1 source contains the same caption twice for S1-S5: a full caption
    # followed by a heading-like duplicate. Keep the full caption only.
    duplicate_headings = {
        "Table S1. Complementary-group results",
        "Table S2. Baseline-recall association",
        "Table S3. Group-conditional MAR summaries",
        "Table S4. Age-conditioned MAR descriptive summaries",
        "Table S5. Type-aware preprocessing comparison",
        "Table S7. Public-health extension results at 20% MCAR under the 80/20 split setting.",
    }
    for p in list(doc.paragraphs):
        if p.text.strip() in duplicate_headings:
            remove_paragraph(p)

    # Replace all Adult rows in S1-S5 with the corrected rerun values. The
    # non-Adult rows remain untouched and are already the canonical values.
    s1 = pd.read_csv(CORR / "complementary_group_summary_corrected_adult.csv")
    s2 = pd.read_csv(CORR / "baseline_recall_summary_corrected_adult.csv")
    s3 = pd.read_csv(CORR / "group_conditional_mar_descriptive_summary_with_corrected_adult.csv")
    s4 = pd.read_csv(CORR / "age_conditional_mar_summary_reconciled_latest.csv")
    s5 = pd.read_csv(CORR / "typeaware_imputation_summary_reconciled_latest.csv")

    def replace_adult(table, formatter):
        for row in table.rows[1:]:
            if row.cells[0].text.strip() != "Adult Income":
                continue
            key = formatter(row)
            src = key[0]
            matched = src
            for j, value in enumerate(key[1], start=1):
                set_cell(row.cells[j], value, highlight=True)

    # Table order is stable in this supplement: S1..S7, S11.
    tables = doc.tables
    assert len(tables) >= 8, len(tables)

    # S1
    for row in tables[0].rows[1:]:
        if row.cells[0].text.strip() == "Adult Income":
            m = s1[(s1.model == row.cells[1].text.strip())].iloc[0]
            vals = [f"{m.mean_fmg_g0:.4f}", f"{m.mean_fmg_g1:.4f}", f"{int(m.positive_fmg_g1_seeds)}/10"]
            for j, v in enumerate(vals, start=2): set_cell(row.cells[j], v, True)
    # S2
    for row in tables[1].rows[1:]:
        if row.cells[0].text.strip() == "Adult Income":
            m = s2[(s2.dataset == "Adult Income") & (s2.model == row.cells[1].text.strip())].iloc[0]
            vals = ["10", f"{m.baseline_recall_min:.4f}–{m.baseline_recall_max:.4f}", f"{m.spearman_rho_baseline_recall_vs_fmg:.3f}"]
            for j, v in enumerate(vals, start=2): set_cell(row.cells[j], v, True)
    # S3
    for row in tables[2].rows[1:]:
        if row.cells[0].text.strip() == "Adult Income":
            model = row.cells[1].text.strip(); rate = float(row.cells[2].text.strip().rstrip('%'))/100
            m = s3[(s3.dataset == "Adult Income") & (s3.model == model) & (s3.rate_nominal == rate)].iloc[0]
            vals = [f"{m.mean_mg_mcar:.4f}", f"{m.mean_mg_mar:.4f}", f"{m.mean_mar_minus_mcar_mg:.4f}", f"{int(m.positive_split_differences)}/10"]
            for j, v in enumerate(vals, start=3): set_cell(row.cells[j], v, True)
    # S4
    for row in tables[3].rows[1:]:
        if row.cells[0].text.strip() == "Adult Income":
            model = row.cells[1].text.strip(); rate = float(row.cells[2].text.strip().rstrip('%'))/100; odds = float(row.cells[3].text.strip())
            m = s4[(s4.dataset == "Adult Income") & (s4.model == model) & (s4.rate_nominal == rate) & (s4.age_odds_ratio_per_sd == odds)].iloc[0]
            vals = [f"{m.mean_age_mar_minus_mcar_fmg:.4f}", f"{int(m.n_positive_differences)}/10"]
            for j, v in enumerate(vals, start=4): set_cell(row.cells[j], v, True)
    # S5
    for row in tables[4].rows[1:]:
        if row.cells[0].text.strip() == "Adult Income":
            strategy = row.cells[1].text.strip()
            m = s5[(s5.dataset == "Adult Income") & (s5.imputation_strategy.str.replace("_numeric", "") == strategy.replace("/", "_"))]
            if m.empty:
                # supplement uses Iterative/mode, KNN/mode, median_mode.
                aliases = {"Iterative/mode": "iterative_numeric_mode", "KNN/mode": "knn_numeric_mode", "median_mode": "median_mode"}
                m = s5[(s5.dataset == "Adult Income") & (s5.imputation_strategy == aliases[strategy])]
            m = m.iloc[0]
            vals = [f"{m.mean_fmg_g0:.4f} ± {m.sd_fmg_g0:.4f}", f"{int(m.positive_fmg_g0_seeds)}/10", f"{m.mean_accuracy_drop:.4f}", f"{m.mean_recall_drop_g0:.4f}"]
            for j, v in enumerate(vals, start=2): set_cell(row.cells[j], v, True)

    # Make the repair idempotent if this script is rerun.
    for p in list(doc.paragraphs):
        if p.text.strip().startswith("Supplementary Table S10."):
            remove_paragraph(p)
    for t in list(doc.tables):
        if t.rows and t.rows[0].cells and t.rows[0].cells[0].text.strip() == "Dataset" and len(t.columns) == 5 and any("ΔTPR" in c.text for c in t.rows[0].cells):
            t._element.getparent().remove(t._element)

    # Add the missing S10 decomposition table immediately before S11.
    s11_caption = next(p for p in doc.paragraphs if p.text.strip().startswith("Supplementary Table S11."))
    s10_caption = doc.add_paragraph("Supplementary Table S10. Aggregate sensitivity–specificity decomposition at 20% MCAR.")
    s10_caption._p.addprevious(s11_caption._p)
    highlight_paragraph(s10_caption)
    rows = [
        ("Adult Income", "0.1288", "−0.0132", "0.0222", "0.1067"),
        ("COMPAS", "0.0222", "0.0280", "0.006?", "−0.0029"),
        ("Taiwan Credit", "0.0825", "−0.0151", "0.0065", "0.0760"),
    ]
    # The manuscript reports COMPAS FMG_all = -0.0029. Do not invent an
    # unreported accuracy-loss value; use an em dash in the supplementary row.
    rows[1] = ("COMPAS", "0.0222", "0.0280", "—", "−0.0029")
    insert_table_before(doc, s11_caption, len(rows)+1, 5,
                        ["Dataset", "ΔTPR_all", "ΔTNR_all", "ΔA", "FMG_all"], rows)

    # S11 was originally captioned after its table. Normalize both new and
    # existing table-caption pairs so each caption precedes its table.
    body = doc.element.body
    p10 = next(p._p for p in doc.paragraphs if p.text.strip().startswith("Supplementary Table S10."))
    p11 = next(p._p for p in doc.paragraphs if p.text.strip().startswith("Supplementary Table S11."))
    t10 = next(t._tbl for t in doc.tables if t.rows and "ΔTPR_all" in t.rows[0].cells[1].text)
    t11 = next(t._tbl for t in doc.tables if t.rows and "ReMasker" in t.rows[0].cells[1].text)
    body.remove(p10); body.remove(p11)
    body.remove(t10); body.remove(t11)
    body.insert(0, p10) if False else None
    # Insert at the original late-supplement position, preserving preceding
    # public-health material while fixing the local order.
    anchor = next((el for el in body if el.tag.endswith('}sectPr')), None)
    pos = list(body).index(anchor) if anchor is not None else len(body)
    body.insert(pos, p10); pos += 1; body.insert(pos, t10); pos += 1
    body.insert(pos, p11); pos += 1; body.insert(pos, t11)

    doc.save(SUPPLEMENT)


def repair_main():
    doc = Document(MANUSCRIPT)
    # Fix [22] with the article DOI and issue/article identifier from the
    # publisher record, rather than the UCI dataset DOI.
    for p in doc.paragraphs:
        if p.text.startswith("[22] Golovenkin"):
            new = ("[22] Golovenkin, S. E., Bac, J., Chervov, A., Mirkes, E. M., Orlova, Y., Barillot, E., Gorban, A. N., & Zinovyev, A. (2020). Trajectories, bifurcations, and pseudo-time in large clinical datasets: Applications to myocardial infarction and diabetes data. GigaScience, 9(11), giaa128. https://doi.org/10.1093/gigascience/giaa128")
            p.text = new
            highlight_paragraph(p)
    # Make the provenance of the separate +0.0169 type-aware result explicit.
    for p in doc.paragraphs:
        if "raw-feature type-aware runner" in p.text and "+0.0169" in p.text:
            old = p.text
            new = old.replace(
                "The separate raw-feature type-aware runner reports +0.0169 for Adult LightGBM under median/mode at 20% MCAR (8/10 positive seeds);",
                "The separate raw-feature type-aware runner, archived in analysis/outputs/typeaware_all_models/, reports +0.0169 for Adult LightGBM under median/mode at 20% MCAR (8/10 positive seeds);"
            )
            new = new.replace("its three-pipeline range remains -0.0121 to 0.0984, and the 24/36 count is unchanged.", "its 36 dataset-model-strategy cell means range from -0.0574 to 0.0984, with 31/36 positive cell means.")
            p.text = new
            highlight_paragraph(p)
    for p in doc.paragraphs:
        if p.text.startswith("All benchmark datasets are publicly available"):
            p.text = p.text.replace("the completed all-model type-aware analysis, the 1,080-row cross-mechanism output", "the completed all-model type-aware analysis (including analysis/outputs/typeaware_all_models/), the 1,080-row cross-mechanism output")
            highlight_paragraph(p)
    doc.save(MANUSCRIPT)


def package_outputs():
    out = ROOT / "analysis" / "outputs" / "typeaware_all_models"
    out.mkdir(parents=True, exist_ok=True)
    src = Path(r"C:\Users\mvk80\Documents\Codex\2026-11-05\search-lacuna\outputs\FMG_final_rectified_release")
    shutil.copy2(src / "typeaware_all_models_per_seed.csv", out / "typeaware_all_models_per_seed.csv")
    shutil.copy2(src / "typeaware_all_models_manifest.json", out / "typeaware_all_models_manifest.json")
    df = pd.read_csv(out / "typeaware_all_models_per_seed.csv")
    summary = (df.groupby(["dataset", "model", "strategy", "rate"], as_index=False)
                 .agg(n_seeds=("seed", "nunique"), mean_fmg=("fmg", "mean"),
                      sd_fmg=("fmg", "std"), positive_seeds=("fmg", lambda x: int((x > 0).sum())),
                      mean_accuracy_loss=("accuracy_loss", "mean"), mean_recall_g0_loss=("recall_g0_loss", "mean")))
    summary.to_csv(out / "typeaware_all_models_summary.csv", index=False)


if __name__ == "__main__":
    package_outputs()
    repair_supplement()
    repair_main()
    print("R12 repairs complete")
