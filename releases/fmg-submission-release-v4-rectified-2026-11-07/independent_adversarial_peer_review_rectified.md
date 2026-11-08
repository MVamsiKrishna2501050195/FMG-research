# Independent Adversarial Peer Review

## Manuscript reviewed

**Title:** Test-Time Feature Missingness Can Hide Recall Loss Behind Stable Accuracy  
**Target:** Scopus-indexed Q2 or Q3 journal  
**Field:** Machine learning, tabular ML, missing data, robustness, and fairness auditing  
**Version:** Rectified DOCX rendered to 21 pages

## Review basis and evidence boundary

This is an independent panel simulation based on the complete rendered manuscript and its tables, equations, figures, references, and declarations. Page numbers below refer to the rendered 21-page version. The review distinguishes:

- **Confirmed from the manuscript:** directly visible wording, equations, tables, or omissions.
- **Potential concern:** plausible reviewer concern that requires checking the code, raw inputs, or publisher metadata.
- **External verification:** limited checks of the cited recent methods were performed against official or primary pages. The connected CitedEvidence manuscript-review service could not be used because the account has no active subscription.

The manuscript is not treated as reproducible merely because it states that code or outputs exist; the claims must be matched to the public release and executable artifacts.

## Executive editorial verdict

**Recommendation: Major Revision.**

The paper is methodologically serious and substantially improved. The frozen-classifier protocol, leakage controls, per-split denominators, multiple datasets, preprocessing sensitivity, native missing-value routing, ReMasker comparison, and explicit limitations are meaningful strengths. The central weakness remains the scientific distinctiveness of FMG. The manuscript now openly admits that FMG contains no information beyond its two component losses and is a reporting convention. That honesty prevents overclaiming, but it also creates the main Q2 rejection risk: a strict reviewer may conclude that the contribution is a renamed difference plus a threshold rule whose outputs can be obtained directly from subgroup recall loss and accuracy loss.

The work is plausibly publishable after major revision in a Q3 venue if the paper is positioned as a reproducible audit protocol and the remaining reproducibility and comparator issues are closed. Q2 suitability is not yet secure because the method has no demonstrated decision utility and the strongest recent missingness-aware adaptation methods are not directly evaluated.

## Five independent reviewer reports

### Reviewer 1 — Senior machine-learning researcher

**Overall assessment:** Important operational problem; technically competent audit; contribution currently reads as a diagnostic convention rather than a new ML method.

**Meaningful problem.** The problem is meaningful. Pages 1, 6, and 14 show that aggregate accuracy can remain relatively stable while subgroup positive-class recall falls. The frozen-classifier, test-only masking setup isolates a real monitoring failure mode.

**Main rejection risk: FMG may be algebraically redundant.** Page 1 states that FMG “does not add mathematical information beyond its two component losses,” and pages 14 and 18 repeat that it is only a paired monitoring convention. Equation (3) is simply `FMG = ΔR - ΔA`. The threshold indicator in Eq. (4) is also directly computable from `ΔR` and `ΔA` without FMG. This is a **confirmed limitation of the claimed contribution**, not a mathematical error.

**Why it matters.** A reviewer can reasonably ask what FMG enables that a table containing subgroup recall loss, aggregate accuracy loss, and a discordance flag does not already provide. The threshold counts on pages 12 and 14 show discordance, but they do not demonstrate analyst utility, earlier detection, fewer false alarms, or a better governance decision.

**Required correction.** Choose one of two defensible positions: (a) reposition the paper explicitly as a reproducible audit/reporting protocol and remove language that implies a novel metric contribution; or (b) add a decision-utility experiment in which blinded monitoring users or a formal alert-decision simulation compare accuracy-only monitoring, direct subgroup-recall monitoring, and the FMG presentation. Report alert sensitivity, specificity, calibration, workload, and decision agreement. Without (b), the paper should not claim that FMG improves monitoring.

**Baseline coverage.** The added FDA, ReMasker, and native-routing analyses are useful, but the FDA implementation is explicitly not an exact reproduction of ATLLM and the manuscript excludes the full LLM-IMPUTE/ATLLM comparison. The official IJCAI-2025 paper does describe LLM-IMPUTE and ATLLM, including augmented training for feature decrement. DiffPuter is also a current ICLR-2025 missing-data imputation baseline. Their omission is a **confirmed scope gap**, not proof that the paper is invalid. At minimum, compare against a computationally feasible subset or make the protocol-level comparison and compute limitation more prominent in the abstract and conclusion.

**ML decision:** Major revision. The benchmark is useful, but novelty and comparator positioning must be tightened.

### Reviewer 2 — Statistical methodology and experimental design specialist

**Overall assessment:** The descriptive design is mostly coherent; inferential language and missingness-mechanism specification need tightening.

**Repeated splits are not population replications.** Pages 5, 14, and 17 correctly state that the ten splits reuse finite observations and that intervals describe split-to-split benchmark stability. This is a strength. However, Table 3 labels the ranges as “95% CI,” while the manuscript says they are percentile bootstrap intervals over ten split-level FMG values. That label can still be read as a population confidence interval. This is a **confirmed presentation/statistical terminology issue**.

**Required correction.** Rename the table column and all prose to “95% percentile bootstrap interval over repeated splits” or “descriptive 95% bootstrap interval.” State the resampling unit in the table note. Do not use these intervals for claims about a population of future datasets.

**Sign-flip tests.** Pages 15–17 use exact two-sided sign-flip tests on ten seed-level paired differences with Holm adjustment. The manuscript calls the results descriptive and notes the finite-data reuse. That qualification is appropriate, but the exchangeability or randomization interpretation should be stated explicitly. Otherwise, “significant” may be interpreted as inferential evidence from ten non-independent resampling splits.

**Required correction.** Replace “significant cells” with “cells with Holm-adjusted descriptive sign-flip p < 0.05,” state the null/randomization assumption, and include a small table of the ten paired differences or a supplementary machine-readable file for every cell.

**Missingness mechanisms are under-specified.** Pages 5, 6, 9, 13, and 17 refer to a group-conditional MAR mechanism, an age-conditioned logistic MAR mechanism, and a value-conditional MNAR mechanism. The manuscript gives nominal rates and age odds ratios but not the complete probability equations, intercept calibration rule, selected feature set, value transformation, clipping, or random-number procedure for the synthetic MNAR mechanism. This is a **confirmed reproducibility omission**. It does not establish that the reported numbers are wrong.

**Required correction.** Give the exact mask-probability equations and pseudocode, identify whether group membership is treated as observed by the missingness process, specify the feature/value used for MNAR, and report the achieved rather than only nominal missingness rate for every dataset and mechanism.

**Aggregation.** The manuscript reports pooled means such as 0.0545 and decomposition values on pages 1, 6, 13, and 14. It should state whether these are unweighted means across dataset-model-split cells, row-weighted means, or means of per-cell means. This is a **potential concern** requiring code/output verification. Different weighting choices can materially change a headline mean.

**Statistical decision:** Major revision, mainly for terminology and full mechanism specification.

### Reviewer 3 — Mathematical and theoretical researcher

**Overall assessment:** The principal algebra is plausible and the repaired proposition is substantially better, but the theory is modest and notation still needs formal cleanup.

**Proposition 1.** On page 13, the manuscript now defines `φ_h(x)=1[h(x)≥t]` and bounds the difference in hard-recall expectations by total variation. Under the conventional total-variation definition and a common input space, this is mathematically sound. The proof is short but valid for a bounded indicator. This is not a confirmed mathematical error.

**Remaining notation concern.** The manuscript does not define the exact total-variation convention. If the authors use the half-L1 convention, the stated bound is correct; if they use an alternative norm-scaled convention, constants can differ. This is a **potential notation concern**.

**Required correction.** Define `TV(P,Q)=sup_B |P(B)-Q(B)| = 1/2 ||P-Q||_1` before Proposition 1. State explicitly that the proposition concerns the hard decision rule and not probability calibration, ranking, or a causal effect.

**Proposition 2.** The accuracy decomposition on page 13 is correct under binary classification, fixed group partition, and the stated condition that only group `g` positive-class recall changes. It is not a general theorem about arbitrary masking changes. The phrase “if masking changes only the positive-class recall of group g” must remain prominent; otherwise a reader could overgeneralize the result. This is a **conditional-validity point**, not an error.

**Theory-to-experiment disconnect.** The propositions are algebraic identities/bounds, not a predictive theory of when FMG will be large. The manuscript does not derive a nontrivial sufficient condition for positive FMG from missingness, feature importance, group prevalence, or classifier geometry. The theoretical section therefore supports interpretation rather than novelty. This is a **confirmed limitation of theoretical contribution**.

**Required correction.** Reframe Section “Theory Extension” as “Interpretive identities and intervention scope,” or add a genuinely predictive proposition with assumptions that can be tested in the experiments. Do not present the current results as a new causal or statistical estimator.

**Mathematical presentation defect.** Page 6 contains the text “aggregate-accuracy loss∈ remains below that same threshold,” which is a stray `∈` and makes Eq. (4)'s surrounding sentence grammatically and mathematically malformed. This is a **confirmed typographical error**.

**Mathematical decision:** Minor-to-major revision depending on how strongly theory is advertised. The existing mathematics is serviceable once its scope is narrowed.

### Reviewer 4 — Algorithmic fairness and robustness specialist

**Overall assessment:** The paper is careful not to call FMG a fairness metric, but the framing still risks fairness overinterpretation and the subgroup design is limited.

**Not a fairness disparity.** Pages 1, 7, 14, and 18 correctly state that FMG compares a subgroup's change across clean/masked conditions, while equal-opportunity-style quantities compare groups within a condition. The empirical fact that comparison-group loss exceeds focus-group loss in many primary cells is important and should remain prominent.

**Residual fairness risk.** The paper uses protected attributes to define g0/g1 and, in one constructed mechanism, to vary masking probability by group. That is legitimate for an audit sensitivity, but the manuscript must make clear that this does not model a deployment mechanism unless the group attribute is actually available to the upstream data-generation process. This is a **potential causal/missingness concern**, not a confirmed error.

**Required correction.** Label the group-conditional experiment “group-conditioned stress sensitivity” throughout, or explicitly define the information set under which group membership is observed. Explain that the mechanism is not evidence about naturally occurring MAR in a deployment.

**Subgroup denominator instability.** The manuscript reports denominator ranges and warns about recall floors, especially UCI Diabetes. This is good. However, the headline threshold counts pool 120 dataset-model-seed evaluations even though subgroup positive-label denominators differ greatly across datasets and cells. A 1% recall-loss flag has different sampling variability at `n+≈316` and `n+≈1,170`, and the pooled alert count is not a fairness-risk prevalence estimate. This is a **confirmed interpretive limitation**.

**Required correction.** Present threshold discordance stratified by dataset and subgroup denominator, or add denominator-weighted sensitivity analyses. State that the 107/100/91/26 counts are counts of benchmark runs, not counts of affected people or deployments.

**Real-world extension.** BRFSS weighted evaluation and naturally incomplete UCI data improve external relevance, but the main claim remains driven by synthetic test-time masks on three standard tabular benchmarks. This is a **potential generalization concern**. The limitations acknowledge it, but the title and abstract should avoid implying broad clinical or financial deployment conclusions.

**Fairness/robustness decision:** Major revision for framing and stratified interpretation; not a rejection on fairness grounds.

### Reviewer 5 — Journal associate editor

**Overall assessment:** The manuscript is coherent and transparent enough for a serious revision decision, but it is not yet an uncomplicated Q2 submission.

**Strengths.** Clear problem statement; frozen-model leakage control; multiple classifier families; primary, exploratory, public-health, type-aware, ReMasker, FDA, and native-routing analyses; explicit limitations; and a candid statement that FMG is not a fairness certificate or causal estimator.

**Editorial risks.**

1. The paper says the new supplementary results are in the final package, but page 18 still points to the older v4 release URL and says the additional files “must be uploaded” before submission. At the time of review, the rectified files are on a new branch/pull request rather than the named v4 release path. This is a **confirmed reproducibility/documentation inconsistency**.
2. The abstract still describes an “operational stress test,” while the manuscript explicitly says there is no evidence of production-monitoring benefit. The phrase is not false, but it needs a modifier such as “benchmark threshold-discordance analysis” to prevent overreading.
3. Table 3's Adult and COMPAS denominator ranges render as `316-` on one line and `-347` on the next, visually resembling a negative number rather than an en-dash range. This is a **confirmed layout/presentation defect** visible on page 7.
4. The paper claims broad coverage of MCAR, MAR, and MNAR while specifying only one mechanism of each structured type. The limitations acknowledge this, but the abstract should call these “constructed sensitivities,” not general validation of missingness mechanisms.

**Editorial decision:** Major Revision. The paper can become a credible Q3 submission after documentation, mechanism, and comparator corrections. For Q2, I would require a stronger demonstration that the audit convention changes a decision or reveals a failure missed by a well-designed existing monitoring baseline.

## Simulated reviewer discussion

### Three most serious rejection risks

1. **FMG is not demonstrably distinct from its components.** This is the most serious risk. The manuscript itself concedes the redundancy. The current threshold counts establish that accuracy-only monitoring can miss subgroup recall loss, but they do not establish that FMG is better than simply reporting subgroup recall loss and accuracy loss together.
2. **Incomplete direct comparison with recent missingness-aware adaptation/recovery.** ReMasker is included and FDA is a useful family-level comparator, but the paper does not directly evaluate ATLLM/LLM-IMPUTE or a diffusion-based imputer such as DiffPuter. The manuscript's own limitation acknowledges this. A strict Q2 reviewer can call the external comparison incomplete.
3. **Reproducibility and mechanism specification.** The manuscript names MAR/MNAR analyses without providing enough equations and calibration details, and its data/code paragraph points to an older release while the new outputs are in a later package. This can prevent independent verification even if the numerical results are correct.

### Scientifically justified criticisms

- FMG's algebraic redundancy and lack of demonstrated decision utility.
- The need to distinguish descriptive repeated-split intervals from population confidence intervals.
- The incomplete specification of constructed MAR/MNAR mechanisms.
- The limited direct baseline coverage relative to recent feature-decrement adaptation and imputation work.
- The risk of treating pooled threshold counts as deployment or fairness evidence.
- The stale or transitional public-release description.

### Mostly presentation concerns

- Stray `∈` in the Eq. (4) sentence on page 6.
- Ambiguous line wrapping of denominator ranges in Table 3 on page 7.
- “Operational stress test” wording in the abstract.
- “Significant cells” wording when the tests are explicitly descriptive.
- Total-variation convention not written explicitly.

### Highest-value additional work

1. Add a compact decision-utility experiment comparing accuracy-only, direct subgroup-recall, and FMG-formatted monitoring under the same 120 benchmark runs. If no human study is feasible, use a predeclared alerting objective and report false-alert/missed-alert tradeoffs, but do not call it human utility.
2. Add one direct, computationally matched missingness-aware training/adaptation baseline, preferably ATLLM's augmented-training principle as an exact or clearly labeled reimplementation, and state why full LLM-IMPUTE or DiffPuter is infeasible if omitted.
3. Publish exact MAR/MNAR generation equations, achieved missingness rates, per-cell outputs, code, and environment metadata.
4. Update the manuscript URL to the merged/tagged rectified release and include a release manifest with hashes for every result used in Tables 4–7.
5. Stratify threshold-discordance results by dataset, model, and subgroup positive-label denominator rather than relying only on pooled counts.

### Is the contribution sufficiently distinctive?

It is distinctive as a named, reproducible audit framing that forces a paired comparison between subgroup recall degradation and aggregate accuracy degradation while preserving denominators and preprocessing provenance. It is **not yet sufficiently distinctive as a new metric or algorithm** because the value is fully reconstructible from the two component losses and the paper does not show decision improvement. The manuscript should choose and consistently defend the audit-protocol positioning.

## Structured editorial report

### A. Overall research quality

Good research hygiene and a meaningful empirical question. The work is stronger than a simple benchmark because it controls leakage, tracks subgroup denominators, tests preprocessing and routing policies, and reports negative or mixed findings. Scientific quality is reduced by under-specified missingness mechanisms and by a contribution whose practical value is asserted more strongly than demonstrated.

### B. Novelty assessment

Moderate as an audit convention; weak-to-moderate as a new scientific measure. The novelty claim is credible only when restricted to the paired audit framing, threshold-discordance reporting, and reproducibility protocol. It is not credible to imply a new information-bearing fairness metric, causal estimand, imputation method, or adaptation algorithm.

### C. Methodological validity

The frozen-classifier test-only masking design is generally valid for the imposed interventions. The main methodological weaknesses are incomplete formalization of MAR/MNAR generation, limited masking-rate coverage for some sensitivities, and uncertainty about whether pooled summaries are weighted consistently.

### D. Mathematical correctness

The repaired Proposition 1 and Proposition 2 are acceptable under their stated conditions. Equation (5) is correct for binary accuracy decomposition. The theory is interpretive rather than predictive. Fix the total-variation convention and the Eq. (4) typographical defect.

### E. Experimental strength

Strong breadth for a CPU-bounded study: three primary datasets, four main classifier families, ten seeds, masking-rate curves, MAR-like sensitivities, an MNAR sensitivity, type-aware preprocessing, a public-health extension, native routing, FDA, ReMasker, and extra operating-point analysis. The comparison set remains incomplete for Q2 because recent feature-decrement adaptation is not directly matched.

### F. Statistical reliability

Adequate for descriptive benchmark analysis, not for population-level inference. The paper mostly acknowledges this correctly. Rename “95% CI,” qualify sign-flip p-values, disclose aggregation rules, and add mechanism-specific achieved-rate and per-cell summaries.

### G. Reproducibility

Potentially strong after the rectified package is merged and tagged. At present, the manuscript's linked release description is transitional and does not yet point directly to the package containing every reported result. The exact MAR/MNAR generator details also need to be in the manuscript or release documentation.

### H. Major revisions required

- Reposition FMG as an audit/reporting convention or add a genuine decision-utility evaluation.
- Add or explicitly justify the absence of a direct recent missingness-aware adaptation baseline.
- Fully specify all constructed MAR/MNAR mechanisms.
- Correct public release links and publish a tagged, hash-verified rectified release.
- Clarify descriptive interval and p-value terminology.
- Stratify or qualify pooled threshold-discordance counts.

### I. Minor revisions required

- Remove the stray `∈` near Eq. (4).
- Improve Table 3 range wrapping.
- Define the total-variation convention.
- Replace “significant cells” with “descriptive Holm-adjusted sign-flip threshold crossings.”
- Replace “operational stress test” in the abstract with “benchmark threshold-discordance analysis.”
- State how headline means are weighted.

### J. Additional experiments required

**For a credible Q3 submission:** exact mechanism documentation, release synchronization, denominator-stratified alert counts, and one matched recent missingness-aware baseline or a transparent compute-based justification.

**For a competitive Q2 submission:** the above plus a decision-utility study or formal operational benchmark showing that FMG-formatted monitoring changes a prespecified decision relative to direct component reporting; preferably an additional independent real-world missingness dataset or a stronger natural-missingness validation.

### K. Publication suitability

- **Q2:** Major Revision, currently borderline and not submission-ready. The central novelty/utility and direct-baseline issues remain material.
- **Q3:** Major Revision, potentially suitable after the listed documentation, wording, and mechanism corrections. Acceptance is not guaranteed.
- **Overall editorial decision:** Major Revision.

## Final recommendation to the authors

Do not add more generic experiments merely to increase the paper's length. The highest-return revision is to make the contribution intellectually honest and operationally testable: present FMG as a standardized audit convention, prove that the protocol is reproducible, compare it against at least one stronger recent missingness-aware method, and demonstrate whether the presentation changes a predeclared monitoring decision. If the decision-utility claim cannot be supported, narrow the title, abstract, and contribution statement accordingly.
