from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
import re

OUT = Path('submission_response_letter_TCSST-26-773.docx')


def h(doc, text, level=1):
    doc.add_paragraph(text, style=f'Heading {level}')


def item(doc, status, location, comment, answer):
    p = doc.add_paragraph()
    match = re.match(r'^(R\d+\.\d+)\s+(.*)$', comment)
    if match:
        r = p.add_run(f'{match.group(1)} {match.group(2)} '); r.bold = True
    else:
        r = p.add_run(f'{comment} '); r.bold = True
    r = p.add_run(f'Status: {status}. '); r.bold = True
    r = p.add_run(f'Location: {location}. '); r.bold = True
    p.add_run(answer)


doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.72); sec.bottom_margin = Inches(0.72)
sec.left_margin = Inches(0.82); sec.right_margin = Inches(0.82)
doc.styles['Normal'].font.name = 'Times New Roman'; doc.styles['Normal'].font.size = Pt(10.2)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('Response to Reviewers'); r.bold = True; r.font.size = Pt(15)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('TCSST-26-773 — Auditing Subgroup Sensitivity to Test-Time Feature Degradation'); r.bold = True
doc.add_paragraph('21 November 2026')
doc.add_paragraph('Dear Editor and Reviewers,')
doc.add_paragraph('Thank you for the detailed review. We revised the manuscript, added the feasible validation analyses, and limited claims where the requested evidence would require a prospective deployment study, human participants, nested cross-validation, or broader compute. New manuscript text is highlighted. Page references below refer to the revised 25-page manuscript; supplementary-table references refer to the six-page supplement.')

h(doc, 'Reviewer 1')
item(doc, 'Clarified', 'Introduction and Discussion, pp. 1–2 and 22', 'R1.1 Novelty and distinct methodology.', 'We now state that FMG is mathematically derived from focus-group recall loss and aggregate-accuracy loss. Its contribution is a prespecified, auditable paired diagnostic for recall-versus-accuracy mismatch under identical test-time masking. We do not claim additional mathematical information, universal superiority, a fairness guarantee, or a deployment guarantee.')
item(doc, 'Clarified', 'Methods and Results, pp. 7 and 9', 'R1.2 Reference threshold.', 'The value 0.01 is described only as a descriptive reference grid point, not as a statistically validated or operational cutoff.')
item(doc, 'Fixed', 'Introduction and Results, pp. 2 and 9', 'R1.3 Interpretation of positive FMG.', 'The manuscript states that positive FMG does not imply greater recall loss in g0 than g1 and reports the component losses. The comparison-group loss exceeds the focus-group loss in 8 of 12 primary pairs.')
item(doc, 'Partial; clarified', 'Results, p. 16; Supplementary Tables S12 and S13, pp. 5–6', 'R1.4 Group selection.', 'S12 is labelled complement-group sensitivity, not an alternative demographic definition. We added a genuine Adult race-group sensitivity: non-White focus versus White comparison. Mean FMG ranges from 0.0679 to 0.1106 across the four models, with 10/10 positive splits for every model. The paper explains that the sensitive attribute is excluded from predictors and is used only for audit partitioning.')
item(doc, 'Clarified', 'Methods, Discussion, and Limitations, pp. 4, 15, and 22', 'R1.5 Synthetic missingness.', 'MCAR, the constructed MAR mechanisms, and the constructed MNAR mechanism are explicitly described as controlled simulations. They do not establish that the mechanisms represent naturally occurring feature degradation.')
item(doc, 'Fixed', 'Methods and Results, pp. 4–5 and 14–15', 'R1.6 Preprocessing.', 'The integer-encoding baseline, training-only scaling, and imputation choices are stated as protocol choices. The type-aware analyses quantify sensitivity rather than claiming preprocessing invariance.')
item(doc, 'Fixed', 'Results and Discussion, pp. 9, 15, and 22', 'R1.7 Dataset variation.', 'The manuscript now identifies when FMG is positive, near zero, or direction-changing and treats this conditional variation as a result, not as a universal effect.')
item(doc, 'Fixed', 'Results and Limitations, pp. 17 and 22', 'R1.8 UCI floor effect.', 'The near-zero UCI values are identified as a baseline-recall floor effect and are not used as evidence of robustness.')
item(doc, 'Limitation', 'Results and Limitations, pp. 18 and 22', 'R1.9 Prospective monitoring.', 'We did not conduct a prospective human-monitoring or deployment study. The screening analysis is retrospective and descriptive; it is not presented as decision-value validation, clinical validation, or proof of operational benefit.')
item(doc, 'Fixed; limitation retained', 'Methods, p. 7; Supplementary Table S10, p. 5', 'R1.10 Repeated splits.', 'We added corrected resampled tests and state that the ten seeds are repeated split realizations from finite datasets, not independent populations.')
item(doc, 'Clarified; limitation retained', 'Results and Limitations, pp. 20–22', 'R1.11 Missingness-aware baselines.', 'ReMasker, feature-decrement augmentation, native routing, and the all-model extension remain matched sensitivity evidence. We do not claim exact ATLLM reproduction or universal superiority. Longer training and broader matched baselines remain future work because the experiments were run CPU-only without a GPU.')
item(doc, 'Fixed; limitation retained', 'Results, p. 14; Supplementary Table S11, pp. 5–6', 'R1.12 SHAP sample size.', 'The COMPAS RF, seed-42 SHAP analysis now uses 100 matched observations per group. It reports clean and median-imputed masked attribution values and descriptive row-level intervals. The median-imputation background shift is described but not corrected for, and the analysis is not causal.')
item(doc, 'Fixed', 'Data Availability and release package, p. 23', 'R1.13 Reproducibility.', 'The release includes scripts, manifests, per-seed outputs, package information, and the exact execution order. The tested package is identified by the v17 release tag.')
item(doc, 'Clarified', 'Theory, p. 8, and Discussion, p. 19', 'R1.14 Equation (7) (reviewer’s Eq. 5).', 'Equation (7) is identified as an aggregate sensitivity-specificity identity, not a direct explanation of subgroup degradation. FMG is computed separately from subgroup recall and aggregate accuracy.')
item(doc, 'Fixed', 'Abstract, Discussion, and Limitations, pp. 1, 19, and 22', 'R1.15 Scope of claims.', 'The results are consistently labelled retrospective benchmark observations. We do not claim fairness improvement, causal effects, clinical value, or deployment safety.')
item(doc, 'Fixed', 'Literature review, p. 2; References, p. 24; Author biographies, p. 25', 'R1.16 Citation quality.', 'Malformed citation expressions, reference ordering, DOI metadata, visual citations, and duplicated visual lead-ins were corrected.')

h(doc, 'Reviewer 2')
item(doc, 'Clarified', 'Theory, pp. 6–8', 'R2.1 Theoretical definition.', 'We explain that FMG contrasts two quantities and is not a between-group disparity decomposition. The prevalence-dependent aggregate identity is interpretive context only.')
item(doc, 'Fixed', 'Results and Discussion, pp. 9 and 19', 'R2.2 Structural limitation.', 'We state that positive FMG can coexist with larger comparison-group recall loss and report both subgroup losses.')
item(doc, 'Fixed', 'Methods and Supplementary denominators, pp. 5 and 7', 'R2.3 Prevalence and subgroup proportions.', 'Per-split positive-label denominators and subgroup proportions are logged and used to bound interpretation. No universal prevalence correction is claimed.')
item(doc, 'Fixed', 'Methods and preprocessing sensitivity, pp. 4–5 and 14–15', 'R2.4 Encoding.', 'The primary encoding is described as a fixed reproducible baseline; the type-aware sensitivity quantifies its dependence.')
item(doc, 'Fixed', 'Results, p. 15; Supplementary Table S5', 'R2.5 Type-aware mechanisms.', 'Median/mode is evaluated at 5%, 10%, and 20% under MCAR, constructed MAR, and constructed MNAR for all four primary models. Five-neighbour KNN/mode and iterative numeric/mode are evaluated at 20% MCAR only. The scope is now stated explicitly.')
item(doc, 'Fixed; limitation retained', 'Statistical Analysis, p. 7; Supplementary Table S10, p. 5', 'R2.6 Corrected inference.', 'The Nadeau--Bengio corrected resampled test uses the ten 70:30 splits and tests mean FMG against zero. After Holm adjustment, all four Adult and all four Taiwan cells are below 0.05; none of the four COMPAS cells are. Full nested cross-validation remains a limitation because the study was run CPU-only without a GPU and the revision budget did not permit a nested rerun.')
item(doc, 'Fixed', 'Methods and Limitations, pp. 4 and 22', 'R2.7 BRFSS weights.', 'Unweighted model fitting, weighted held-out evaluation, and the compact training-weight sensitivity check are distinguished as descriptive components.')
item(doc, 'Fixed', 'Results and Limitations, pp. 17 and 22', 'R2.8 UCI floor effect.', 'The floor effect is stated explicitly and is excluded from robustness claims.')
item(doc, 'Limitation', 'Results and Limitations, pp. 21–22', 'R2.9 Native-routing divergence.', 'The LightGBM/XGBoost divergence remains unresolved. We do not infer a mechanism without a feature-path analysis.')
item(doc, 'Fixed', 'Results, pp. 14–15', 'R2.10 Calibration and ranking.', 'Brier score, ECE, and PR-AUC were added from archived row-level probabilities as supporting sensitivity analyses, not deployment validation.')
item(doc, 'Limitation', 'Results and Limitations, pp. 20–22', 'R2.11 ReMasker parity.', 'The matched validation is limited to the 50-epoch CPU configuration. Longer training, capacity tuning, and broader matched adaptation comparisons remain future work because the experiments were CPU-only without a GPU.')
item(doc, 'Fixed; limitation retained', 'Results, p. 14; Supplementary Table S11, pp. 5–6', 'R2.12 SHAP sample size.', 'The added run is COMPAS Random Forest, seed 42 only, with 100 matched observations per group. It reports the median-imputed background shift but does not correct for that shift or infer causality.')

h(doc, 'Editor minor comments')
item(doc, 'Fixed', 'Throughout; especially pp. 9–16', 'Table and figure numbering and in-text citations.', 'Tables 1–6 and Figures 1–9 are numbered, captioned, and cited in the body. Captions were shortened, and duplicate visual lead-ins were removed.')
item(doc, 'Fixed', 'Methods and Theory, pp. 6–8 and 19–20', 'Equation numbering.', 'Displayed mathematical equations are numbered sequentially and cited where used.')
item(doc, 'Fixed', 'References, p. 24', 'Bracketed reference numbering.', 'Bracketed numerical citations were restored and checked against the reference list.')
item(doc, 'Fixed with justification', 'Literature review, p. 2; References, p. 24; Author biographies, p. 25', 'Recent references and retained pre-2020 sources.', 'Recent publications were added where they support current missingness, fairness, adaptation, and imputation claims. The four retained pre-2020 sources are kept because they are foundational method or original dataset citations; replacing those primary sources would reduce provenance rather than improve currency.')
item(doc, 'Fixed', 'References, p. 24', 'ArXiv-only references.', 'The cited ReMasker and DiffPuter works use their published conference or official proceedings/open-review records rather than arXiv-only metadata. DOI and venue metadata were checked.')

h(doc, 'Remaining limitations')
doc.add_paragraph('The revision does not claim prospective human-monitoring benefit, nested-cross-validation population inference, exact reproduction of every missingness-aware adaptation method, correction of the SHAP imputation-background shift, or a causal explanation of attribution changes. These are stated as future work or limitations rather than left implicit.')
doc.add_paragraph('We appreciate the reviewers’ suggestions, which improved the scope, statistical qualification, demographic sensitivity analysis, and reproducibility of the manuscript.')
doc.add_paragraph('Sincerely,')
doc.add_paragraph('The authors')
doc.save(OUT)
print(OUT)
