# Independent Adversarial Peer Review

## Scope and evidence

This review examines the rendered 21-page manuscript `JAISCR_FMG_submission_q2_q3_ready_native_baseline.docx`, the completed type-aware and native-missingness CSV/JSON outputs, and the public FMG repository. The repository verification script was executed locally and passed its saved-output checks: 12 primary summaries, FMG identities, subgroup denominators, bootstrap rows, 90 legacy type-aware rows, and nine numbered figures. It does not retrain the models.

The specialist manuscript-review service was unavailable because the connected CitedEvidence account has no active subscription. Literature checks therefore use primary publisher or conference sources and the manuscript's cited bibliography.

## Executive editorial assessment

Decision: **Major Revision**.

The research problem is meaningful and the frozen-classifier audit is methodologically substantial. The paper is potentially suitable for a Q3 journal after correction and a realistic Q2 attempt after stronger release harmonization and contribution positioning. It is not yet safe to submit because the final manuscript and public release are not synchronized, and one statistical description is contradicted by the actual repository artifact.

The most serious confirmed issue is on page 14 under **Uncertainty and validation**: the manuscript says it uses “patient-level bootstrap resampling with masking randomization nested inside each bootstrap sample.” The repository artifact `fmg_bootstrap_ci_primary.csv` and its regeneration script instead describe a **seed-level bootstrap over 10 repeated-split FMG values**, with the interval scope explicitly marked “descriptive repeated-seed benchmark.” This must be corrected or the claimed nested bootstrap must be independently generated and archived.

The second major issue is release integrity. The latest manuscript reports the 360-row all-model type-aware analysis, the 60-row native XGBoost/LightGBM baseline, the FDA comparator positioning, and the theory/stress-test material. The public v4 repository README still says the type-aware sensitivity is limited to RF, three datasets, ten seeds, and 20% MCAR, and does not list the new native-baseline, all-model type-aware, theory, or stress-test artifacts. The final manuscript therefore currently contains results that an independent reader cannot reproduce from the cited release.

## Reviewer 1: Senior machine-learning researcher

### Assessment

The problem is important: test-time feature loss can alter subgroup recall without an obvious change in aggregate accuracy. The primary frozen-classifier design is appropriate for isolating inference-time masking effects, and the paper includes diverse model families and several stress tests.

The gap is only partially genuine. Test-time feature decrement and missing-feature adaptation are now active research topics. Cheng et al. study fully test-time feature decrement and propose LLM-IMPUTE and ATLLM at IJCAI 2025; Zhou et al. study fully test-time adaptation for tabular data at AAAI 2025. ReMasker and DiffPuter address missing-value recovery. The defensible gap is narrower: a reproducible subgroup-recall-versus-aggregate-accuracy audit of a frozen classifier across controlled test-time masks, with explicit subgroup denominators and recovery-pipeline sensitivity.

### Main criticism

FMG is not a new predictive algorithm and is algebraically derived from two reported losses. The paper correctly admits this, but the contribution remains vulnerable if called a “metric” rather than a reporting convention or diagnostic framework. The threshold-discordance results establish that direct subgroup recall can change a monitoring decision; they do not show that FMG contains information unavailable from the two components.

### Required correction

Keep the current bounded framing: FMG is a paired audit convention, not a fairness metric, causal estimator, or superior detector. Make the operational example the central practical contribution and state that direct subgroup recall remains the primary safety signal.

## Reviewer 2: Statistical methodology and experimental design specialist

### Confirmed issue

Page 14 claims patient-level nested bootstrap validation, but the repository's bootstrap file and script resample the ten seed-level FMG values. The manuscript's own page 5 says the intervals are descriptive repeated-split summaries. These cannot both be presented as the same uncertainty procedure.

### Statistical concern

The ten seeds are repeated splits of the same finite data, not independent populations. The manuscript mostly acknowledges this correctly. However, exact sign-flip p-values and Holm-adjusted p-values in the FDA and ReMasker comparisons can still be read as inferential evidence despite the repeated-split dependence. They should be labeled exploratory paired randomization summaries, with effect sizes and intervals, not evidence of population-level superiority.

### Additional concerns

- The 0.01 threshold is descriptive and not decision-calibrated. It must not be described as a practical significance threshold.
- Bootstrap intervals over ten seed-level values are unstable and should be presented as benchmark variability intervals, not confidence intervals for a population parameter.
- The FDA and ReMasker comparisons use different recovery/training procedures and should not be pooled into one superiority ranking.

### Required correction

Correct the bootstrap description, retain the explicit descriptive scope, and make the statistical unit and estimand visible in every comparison table.

## Reviewer 3: Mathematical and theoretical researcher

### Definitions

Equations (1)--(5) are essentially correct under the stated binary-classification definitions. If clean-minus-masked losses are used, each component lies in [-1,1] and FMG lies in [-2,2]. The accuracy decomposition in Equation (5) is correct when prevalence and specificity are defined on the same held-out cohort.

### Confirmed notation issue

Page 13 defines a frozen classifier score `h` in [0,1] but defines the intervention effect using positive-class recall. Recall is the expectation of a hard indicator, not generally the expectation of an arbitrary probability score. The bound is valid if `h` is explicitly defined as the hard prediction indicator `1{score >= threshold}` or if the estimand is changed to mean score rather than recall.

### Proposition 2

The visibility result is correct under its narrow assumption that masking changes only the positive-class recall of the selected group. The notation `pi_g1 = Pr(G=g,Y=1)` must remain clearly identified as a joint population mass, not subgroup prevalence. The result does not imply that the group is uniquely harmed; the manuscript correctly compares g0 and g1 separately.

### Total variation

The total-variation bound is a valid expectation bound only when the clean and transformed inputs are represented on a common measurable input space and the classifier output is bounded. It is a loose validity bound, not a predictive explanation of observed FMG. The finite numerical check does not prove the empirical bound for all data-generating processes.

### Required correction

Define `h` as the hard classifier indicator for the recall proposition, or separate the score-based bound from the recall-based estimand. Clarify that the TV result is a formal upper bound and not an empirical causal or explanatory model.

## Reviewer 4: Algorithmic fairness and robustness specialist

### Strengths

The manuscript is careful not to equate FMG with Equal Opportunity or a between-group disparity. The fact that g0 recall loss exceeds g1 recall loss in only 4/12 primary pairs is important and should remain prominent.

### Concerns

- The selected focus groups and labels are dataset-specific, so the audit is not a universal fairness test.
- MCAR is simulated; the group-conditional MAR mechanism is constructed; the value-conditional MNAR experiment is synthetic. None identifies natural missingness mechanisms.
- The BRFSS and UCI extension improves external validity, but it remains retrospective benchmark analysis, not deployment or clinical validation.
- The observed native-routing results show that preprocessing and model-family choices can reverse FMG direction. This strengthens the paper as an audit, but weakens any universal interpretation.

### Required correction

State that FMG is a conditional subgroup-robustness diagnostic. Require the group definition, missingness policy, classifier, preprocessing, and operating point to be reported with every FMG value.

## Reviewer 5: Journal associate editor

### Editorial view

The manuscript has enough substance for serious review, but it is too easy for a reviewer to find a mismatch between the cited release and the latest claims. That alone can cause rejection on reproducibility grounds.

The current title is attention-catching but slightly causal-sounding. A safer title is:

**Auditing Subgroup Recall Under Test-Time Feature Missingness in Tabular Classifiers**

The present title can be retained if the abstract and conclusion continue to state that the result is conditional and not a deployment-safety guarantee.

### Required editorial action

Before submission, publish a new release containing the final manuscript, all new CSV/JSON outputs, scripts, manifests, notebook versions, exact environment information, and a README that distinguishes historical v4 results from the new final analysis.

## Simulated reviewer discussion

### Three most serious rejection risks

1. **Reproducibility mismatch:** latest manuscript claims are not all present in the public v4 release.
2. **Uncertainty misdescription:** patient-level nested bootstrap language conflicts with the actual seed-level bootstrap artifact.
3. **Contribution distinctiveness:** FMG is algebraically redundant with its components; its value must remain operational/reporting value, not new information.

### Scientifically justified criticisms

- Bootstrap description and repository mismatch are confirmed defects.
- The score-versus-hard-recall notation requires correction.
- Exact ATLLM/LLM-IMPUTE reproduction is absent; the current FDA comparator is only a family-level analogue.
- Natural MAR/MNAR generalization is not established.

### Mainly presentation concerns

- The paper is dense and could reduce repeated numeric narration.
- Table 3 denominator ranges need typography cleanup; the rendered `316- -347` style should become a clear en dash range.
- The new native-baseline and all-model type-aware results need compact tables rather than prose alone.

### Highest-value additional work

1. Harmonize the public release and rerun the repository verification from that release.
2. Correct the bootstrap and theory language.
3. Add compact supplementary tables for the 360-row type-aware and 60-row native-baseline analyses.
4. Exact ATLLM reproduction is optional and expensive; it is not required if the paper explicitly limits the claim to family-level missingness-aware training/recovery sensitivity.

## Research-gap and originality matrix

| Claim | Status | Adversarial conclusion |
|---|---|---|
| Test-time feature decrement is under-audited | Partially addressed | Recent IJCAI and AAAI work directly studies feature decrement and test-time adaptation. |
| Aggregate accuracy can hide subgroup recall loss | Partially addressed | Adjacent missingness/fairness and robustness work exists; the paper's frozen paired audit is more specific. |
| FMG is a new mathematical metric | Already addressed by algebra | It is a derived contrast of two losses and should not be sold as mathematically new. |
| FMG is a reproducible audit convention | Potentially novel | This is the strongest defensible contribution if the release is synchronized. |
| Missingness and preprocessing alter the audit signal | Established and extended | Prior imputation studies support the premise; the paper adds multi-dataset subgroup-recall auditing. |
| MCAR/MAR/MNAR generalization | Partially addressed | Only specified simulated mechanisms are tested; natural mechanism claims are unsupported. |
| FMG improves operational decisions | Unverified | The finite synthetic stress test shows a thresholding pattern, not human or deployment utility. |

## Numerical and repository audit

### Verified locally

- The primary repository verification script passed saved-output checks.
- The native baseline has 60/60 rows, no duplicates, and no missing FMG values.
- The all-model type-aware output has 360/360 rows, no duplicates, and no missing FMG values.
- The rendered manuscript has 21 pages and no visible layout defects in the final QA pass.

### Confirmed release problem

The public v4 repository README states that type-aware results remain limited to RF and lists the v4 archive as authoritative. It does not contain the new all-model type-aware output, native-baseline output, FDA positioning artifacts, or the final theory/stress-test implementation. The latest manuscript therefore cannot currently be independently regenerated from its cited public release.

## Required revision checklist

### Major revisions

- Replace the patient-level nested-bootstrap claim with the actual seed-level descriptive bootstrap, or generate and archive the claimed nested bootstrap.
- Upload the final manuscript, all new outputs, scripts, notebooks, manifests, and environment files to a new public release.
- Correct the `h` definition in Proposition 1.
- Add supplementary tables for the type-aware all-model and native-routing analyses.
- Label FDA, ReMasker, and native routing as sensitivity/comparator families, not universal mitigation methods.
- Make the abstract, conclusion, README, and repository manifest agree exactly.

### Minor revisions

- Fix rendered denominator ranges and typography.
- Reduce repeated claims that FMG is not a fairness disparity; state it once prominently and refer back to it.
- Keep the 0.01 value explicitly descriptive everywhere.
- Verify all final reference metadata and official URLs.

## Publication decision

**Q3:** Major Revision. After the release, bootstrap, and notation corrections, the manuscript is a credible Q3 submission.

**Q2:** Major Revision / borderline. The research design and breadth are competitive, but Q2 reviewers may still require a more distinctive contribution or exact recent adaptation baseline. The paper should be submitted to Q2 only after the reproducibility mismatch and confirmed statistical-description error are fixed.

**Final simulated editorial decision: Major Revision, not Accept.**

## Closest literature

- Cheng, Jia, Yu, Zhou, and Guo, “Fully Test-Time Adaptation for Feature Decrement in Tabular Data,” IJCAI 2025, DOI 10.24963/IJCAI.2025/550: directly studies feature decrement and proposes LLM-IMPUTE and ATLLM.
- Zhou, Yu, Guo, and Li, “Fully Test-time Adaptation for Tabular Data,” AAAI 2025, DOI 10.1609/aaai.v39i21.34466: studies test-time adaptation under tabular distribution shifts.
- Du, Melis, and Wang, “ReMasker: Imputing Tabular Data with Masked Autoencoding,” ICLR 2024: missing-value imputation front-end.
- Zhang, Fang, Wu, and Yu, “DiffPuter: Empowering Diffusion Models for Missing Data Imputation,” ICLR 2025: diffusion/EM imputation front-end.
- Min, Asif, and Vaidya, “Exploring the Inequitable Impact of Data Missingness on Fairness in Machine Learning,” IEEE Intelligent Systems 2025, DOI 10.1109/MIS.2025.3549484: directly relevant fairness/missingness analysis.
- Gabr, Helmy, and Elzanfaly, “Effect of Missing Data Types and Imputation Methods on Supervised Classifiers: An Evaluation Study,” Big Data and Cognitive Computing 2023, DOI 10.3390/bdcc7010055: relevant multi-model missing-data evaluation showing why accuracy alone is insufficient.

These studies make the gap narrower, not invalid. The paper's defensible contribution is the frozen-classifier, subgroup-recall/aggregate-accuracy audit plus explicit denominator and pipeline sensitivity, not a new adaptation algorithm.
