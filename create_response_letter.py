from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

OUT = Path('submission_response_letter_TCSST-26-773.docx')

def heading(doc, text, level=1):
    p = doc.add_paragraph(style=f'Heading {level}')
    p.add_run(text)

def response(doc, label, text):
    p = doc.add_paragraph()
    r = p.add_run(label + ' ')
    r.bold = True
    p.add_run(text)

doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.75); sec.bottom_margin = Inches(0.75)
sec.left_margin = Inches(0.85); sec.right_margin = Inches(0.85)
doc.styles['Normal'].font.name = 'Times New Roman'
doc.styles['Normal'].font.size = Pt(10.5)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('Response to Reviewers'); r.bold = True; r.font.size = Pt(15)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('TCSST-26-773 — Auditing Subgroup Sensitivity to Test-Time Feature Degradation'); r.bold = True
doc.add_paragraph('21 November 2026')
doc.add_paragraph('Dear Editor and Reviewers,')
doc.add_paragraph('Thank you for the detailed review of our manuscript. We made the requested low-cost analyses that could be completed without changing the frozen-classifier protocol, and we moved the full numerical results to the supplementary material to preserve the 25-page main-manuscript limit. New manuscript text is highlighted. We also narrowed claims where the requested evidence would require prospective deployment, human participants, broader matched comparisons, or substantially larger compute.')

heading(doc, 'Summary of substantive changes')
for item in [
    'Supplementary Table S10 reports Nadeau--Bengio corrected resampled t-tests of mean FMG against zero for all 12 primary dataset-model cells. The correction uses the ten fixed 70:30 splits and the factor 1/k + n_test/n_train. These tests address repeated-split dependence but do not provide external population inference or replace nested cross-validation.',
    'Supplementary Table S11 reports an alternative focus-group definition in which g1 is treated as the focus group using the same archived predictions, masks, and seeds. Mean g1 FMG ranges from 0.0928 to 0.1189 on Adult Income, -0.0349 to -0.0113 on COMPAS, and 0.0742 to 0.0856 on Taiwan Credit.',
    'Supplementary Table S12 reports COMPAS Random Forest SHAP sensitivity using 100 matched observations per group rather than 20. priors_count remains the largest mean absolute SHAP shift in both groups. The analysis remains descriptive and non-causal.',
    'The missing paragraph break before “All-model extension” was restored. The main manuscript remains 25 pages, and the supplement is supplied with the submission package.',
]:
    doc.add_paragraph(item, style='List Bullet')

heading(doc, 'Reviewer 1')
response(doc, 'R1.1 Novelty and distinct methodology.', 'We clarified that FMG is mathematically derived from focus-group recall loss and aggregate-accuracy loss. Its contribution is operational packaging: a prespecified paired diagnostic that identifies a recall-versus-accuracy mismatch under identical test-time masking. We no longer present FMG as adding information beyond its components, as a superior fairness metric, or as a fairness guarantee.')
response(doc, 'R1.2 Reference threshold.', 'We now describe 0.01 only as a descriptive reference grid point, not a validated operational threshold or significance boundary. Threshold-dependent claims are explicitly limited.')
response(doc, 'R1.3 Interpretation of positive FMG.', 'We retain the distinction throughout the manuscript: positive FMG does not imply greater recall loss in g0 than g1. The manuscript reports both subgroup losses and notes that g1 loss exceeds g0 loss in 8 of 12 primary pairs.')
response(doc, 'R1.4 Group selection.', 'We added the alternative g1-as-focus-group sensitivity in Supplementary Table S11 and a highlighted summary in the Results. The sign and scale are dataset-dependent: Adult and Taiwan remain positive, whereas COMPAS is negative for all four models under this alternative definition.')
response(doc, 'R1.5 Synthetic missingness.', 'We state explicitly that MCAR, MAR, and MNAR mechanisms are controlled simulations and do not establish representativeness of naturally occurring degradation. The independent natural-missingness check is also described as confounded and non-causal.')
response(doc, 'R1.6 Preprocessing.', 'We retained the type-aware sensitivity analysis and narrowed its interpretation to preprocessing dependence under the evaluated pipelines. Integer encoding, training-only scaling, and imputation are described as protocol choices and limitations.')
response(doc, 'R1.7 Cross-dataset variation.', 'We consolidated the conditions under which FMG is positive, near zero, or direction-changing. The Discussion treats this variation as a central conditional finding rather than evidence of universal behavior.')
response(doc, 'R1.8 UCI floor effect.', 'The UCI result is explicitly identified as a floor-effect limitation. Near-zero clean focus-group recall is not used as evidence of robustness.')
response(doc, 'R1.9 Operational value.', 'We did not conduct a prospective human-monitoring or deployment study. The retrospective screening analysis is labeled as descriptive rule comparison, not decision value, clinical validation, or deployment benefit. A prospective human study remains future work.')
response(doc, 'R1.10 Repeated splits.', 'We report corrected resampled tests in Supplementary Table S10 and state that repeated seeds are repeated split realizations from finite datasets, not independent populations. Claims remain benchmark-stability claims.')
response(doc, 'R1.11 Missingness-aware baselines.', 'The manuscript retains ReMasker, feature-decrement augmentation, native routing, and the all-model extension as matched sensitivity evidence. We do not claim exact ATLLM reproduction or universal superiority. Longer training and broader matched adaptation comparisons remain future work because of the available compute budget.')
response(doc, 'R1.12 SHAP.', 'We increased the COMPAS RF matched sample to 100 observations per group and report the full feature-level table in Supplementary Table S12. We continue to state that SHAP changes do not establish causes or causal mechanisms.')
response(doc, 'R1.13 Reproducibility.', 'The release includes scripts, manifests, per-seed outputs, package information, input hashes where available, and the exact execution order. The corrected release tag identifies the tested package.')
response(doc, 'R1.14 Equation (5).', 'We clarified that the aggregate sensitivity-specificity decomposition is an algebraic identity for aggregate accuracy and is not a direct explanation of subgroup degradation. FMG is computed separately from subgroup recall and aggregate accuracy.')
response(doc, 'R1.15 Scope of claims.', 'We consistently label the results as retrospective benchmark observations. We do not claim fairness improvement, causal effect, clinical benefit, or deployment safety.')
response(doc, 'R1.16 References and citation formatting.', 'We corrected malformed citation expressions, reference ordering, DOI metadata, table and figure citations, and duplicated visual lead-ins. The revised submission uses one external caption per visual object.')

heading(doc, 'Reviewer 2')
response(doc, 'R2.1 Theoretical definition.', 'We clarified that FMG is a contrast between two different quantities and is not a decomposition of between-group disparity. The prevalence-dependent aggregate identity is presented only as context for interpreting aggregate accuracy.')
response(doc, 'R2.2 Structural limitation.', 'We state that positive FMG can coexist with larger comparison-group recall loss. This is why all interpretations report g0 and g1 recall losses separately.')
response(doc, 'R2.3 Prevalence and subgroup proportions.', 'The manuscript reports subgroup positive-label denominators and ranges from the archived per-split logs. The prevalence scaling relationship is presented as a bound and interpretive limitation, not as a universal correction.')
response(doc, 'R2.4 Encoding.', 'The primary pipeline is explicitly described as a fixed, reproducible baseline rather than a claim that ordinal encoding is semantically correct for nominal variables. Type-aware results quantify sensitivity to this choice.')
response(doc, 'R2.5 Type-aware mechanisms.', 'The completed sensitivity package covers the evaluated rates and constructed mechanisms where computationally feasible. We retain the limitation that no finite set of mechanisms establishes preprocessing invariance under all natural missingness processes.')
response(doc, 'R2.6 Corrected inference.', 'Supplementary Table S10 adds the Nadeau--Bengio corrected resampled test. Full nested cross-validation remains outside this revision because the available compute does not support a full nested rerun before revision. We state this limitation explicitly.')
response(doc, 'R2.7 BRFSS weights.', 'The manuscript distinguishes unweighted model fitting from weighted held-out evaluation and reports the compact training-weight sensitivity check as descriptive evidence.')
response(doc, 'R2.8 UCI floor effect.', 'We retain the explicit floor-effect warning and do not use the UCI near-zero FMG values to support robustness claims.')
response(doc, 'R2.9 Native-routing divergence.', 'The divergence between LightGBM and XGBoost is reported as unresolved classifier-policy interaction. We do not infer a mechanism without a feature-path analysis.')
response(doc, 'R2.10 Calibration and ranking.', 'Brier score, equal-width ECE, and average precision/PR-AUC were added as sensitivity analyses using the archived row-level probabilities. They are reported as supporting evidence and not as causal or deployment validation.')
response(doc, 'R2.11 ReMasker parity.', 'The manuscript reports the matched 50-epoch CPU validation and its resource limitation. Longer training, capacity tuning, and broader adaptation baselines are named as future work rather than treated as completed evidence.')
response(doc, 'R2.12 SHAP sample size.', 'The COMPAS RF SHAP analysis now uses 100 matched observations per group, reports clean and masked mean absolute values, row-level shifts, and descriptive intervals in Supplementary Table S12. The masking and imputation protocol is stated, and causal interpretation is excluded.')

heading(doc, 'Remaining limitations')
doc.add_paragraph('The revision does not claim prospective human-monitoring benefit, clinical validation, population-level inferential validity, exact reproduction of every missingness-aware adaptation method, or a causal explanation of SHAP changes. These limitations arise from the retrospective benchmark design, finite public datasets, fixed split protocol, and light-compute revision environment. We identify prospective human-in-the-loop evaluation, nested cross-validation, longer and tuned recovery baselines, and mechanism-specific native-routing analysis as future work.')
doc.add_paragraph('We appreciate the reviewers’ suggestions, which helped us narrow the claims and make the audit protocol more transparent and reproducible.')
doc.add_paragraph('Sincerely,')
doc.add_paragraph('The authors')
doc.save(OUT)
print(OUT)
