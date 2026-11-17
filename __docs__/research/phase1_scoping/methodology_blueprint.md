# Methodology Blueprint

Phase 1 · `research_architect_agent` · 2026-11-17 · inputs: `rq_brief.md`, `plan.md` §2–§4, `configs/stage3_h1.json`, `src/exchange_ml/{metrics,instances}.py`.

## Research Paradigm

**Selected:** Positivist (quantitative, confirmatory with an exploratory decomposition stage).
**Justification:** The RQ is comparative and measurable: scores and risks are computed deterministically from stored predictions and ground truth; the claim is a difference in a ranking-quality statistic. No interpretive component.

## Method

**Type:** Quantitative.
**Specific method:** Retrospective benchmarking of failure-detection scores with a two-stage design — (i) exploratory decomposition on the diagnosis set (validation, 40 cases) that generated the hypothesis; (ii) pre-registered confirmatory comparison on a held-out set the models and the analysts have not seen.
**Justification:** The hypothesis (partition term carries the signal; v1 hid it) was formed *after* seeing validation results. The only honest test is a frozen protocol applied to new cases. This also answers the main objection the first two H1 runs attracted — tuning on the analysis set.

## Data Strategy

**Data type:** Secondary (public PENGWIN 2026 Task 1 CT + instance labels) plus existing model outputs produced in this project.

| Set | Cases | Units (case × bone) | Role | Seen by members? |
|---|---|---|---|---|
| Validation (diagnosis) | 40 (20 pelvic, 20 femur) | 80 | Sub-questions 1, 3; hypothesis generation; reported as-is | Yes (validation only; not trained on) |
| Held-out (confirmation) | 100 from the 210 pool cases outside trajectory-1's initial set, seeded, stratified 50 pelvic / 50 femur | 200 | Sub-questions 2, 3 | No |
| Test | 50 | — | Sealed | No |

**Sampling rationale for 100:** fits the GPU budget (≈ 8–56 GPU-h) and schedule; gives 200 units and 100 independent clusters for the bootstrap. **WP1 deliverable (DA-2):** before the case list is committed, compute the case-cluster bootstrap CI half-width of the primary AURC difference on the validation set and project it to n = 100 and n = 210. Pre-declared rule: if the projected half-width at n = 100 exceeds the validation effect (AURC 0.798 − 0.739 = 0.059, a proxy for v2 vs Jaccard), raise n to 210 (≈ 17–117 GPU-h) or, if the schedule cannot absorb that, record the analysis-paper framing (`plan.md` §3.3) as the expected outcome *before* inference. The projection and the decision are written into the v2 config.

**Risk sources per unit:**
- Primary: mean over the three members of each member's own per-bone risk (same models produce score and risk).
- Secondary: Stage 2 single-model per-bone risk (the v1 source) — kept so that condition C3(a) can be shown rather than asserted.

**Score inputs:** member instance maps after the frozen cleanup (26-connectivity, < 1000 mm³ pruned, no morphology), Stage-B foreground probabilities for entropy, shared ROI per bone = union of the three cleaned member foregrounds within that bone's ID range.

**Time frame:** member outputs Oct 2026 (validation) and Dec 2026 (held-out).

## Analytical Framework

### A. Decomposition (Sub-question 1, exploratory, validation set)
1. For each pair (i, j) and bone: presence term `|F_i △ F_j| / |U|`, partition term `(|C| / |U|) · NVI(P_i|_C, P_j|_C)`, under both normalisations `VI / log|C|` (v1, frozen) and `VI / H(P,Q)` (v2).
2. Report distribution (median, IQR, max) of each term and its share of the sum; Spearman ρ between each term and foreground Jaccard.
3. Analytical bound: for k fragments on the shared foreground, `VI ≤ 2 log k`; with `log|C| ≈ 12–14` this caps the v1 partition term at ≈ `2 log k / log|C|` — tabulate for k = 2…12 and compare with observed values.

### B. Confirmatory comparison (Sub-question 2, pre-registered, held-out set)
- **Primary endpoint:** AURC of normalised merge+split risk (lower is better) for FA-IPD v2 versus (a) foreground Jaccard disagreement and (b) mean entropy.
- **Decision rule:** both differences ≤ −0.02 AURC (minimal meaningful effect) with 95 % bootstrap CIs (10 000 resamples, resampling cases, carrying their bones as a cluster) excluding 0. *(DA-1/DA-missing: effect floor added.)*
- **Both risk sources for every row** — primary (member-mean) and secondary (Stage 2 single model); the C2 conclusion is reported as robust only if it holds under both. *(DA-3.)*
- **Wording:** v2 is "specified and frozen before held-out scoring", never "a priori"; the paper states the sequence v1 fail → diagnosis → v2 freeze → held-out test. *(DA-1.)*
- **Secondary endpoints:** `1 − F1` AURC; missing-fragment (recall < 1) AURC; top-quartile enrichment (prevalence fixed at 25 %); Spearman ρ and partial Spearman controlling for fragment count, bone volume, minimum-fragment volume; per-scan-type strata.
- **Rows reported:** entropy (mean, p95), pairwise Dice, Jaccard, union-NVI, count variance, Hungarian, FA-IPD v1, presence term alone, partition term (v2) alone, FA-IPD v2.
- **Ties:** rank-average; AURC via trapezoid over coverage 1/n … 1.
- All of the above is written into `configs/stage3_h1_v2.json` and committed (hash in paper) before held-out inference starts.

### C. Evaluation conditions (Sub-question 3)
| Condition | Manipulation | Statistic reported |
|---|---|---|
| (a) risk-source mismatch | primary vs secondary risk source | change in AURC and in score ranking |
| (b) prevalence saturation | failure = "any merge/split" (prev. 0.925) vs top quartile (0.25) | enrichment ceiling and achieved enrichment |
| (c) annotation scope | case-level ROI (v1) vs per-bone units (v2) | fraction of disagreement on unlabelled voxels; AURC change |
| (d) matching-rule blindness | IoU ≥ 0.1 instance F1 vs GT-volume-fraction-unmatched per bone | count of units with F1 = 1 and volume fraction < 0.9 |

Condition (d) adds one pre-registered secondary risk (unmatched GT volume fraction per bone); the frozen IoU rule itself is unchanged.

### D. Mechanism check (H4, CPU)
Apply the existing `perturbations.py` merge / split / missing-fragment / boundary perturbations to GT instance maps; report v1 and v2 partition-term response curves. Expected: v2 responds to merge/split, both respond to missing, neither to boundary-only.

**Tools:** existing `exchange_ml` package, NumPy/SciPy; bootstrap and partial-correlation code added with unit tests in WP1.

## Validity Criteria

| Criterion | Strategy |
|---|---|
| Internal validity (no post-hoc tuning) | v1 frozen and reported unchanged; v2 config committed with hash before held-out scoring; one primary normalisation; decision rule fixed. |
| Internal validity (score/risk coherence) | Primary risk from the same models that produce the score; mismatched source kept only as condition (a). |
| Construct validity (structural failure) | Three structural risks (merge+split, F1, missing-fragment) plus `1 − Dice` control; H4 perturbations show the score responds to the intended error types. |
| Statistical conclusion validity | Case-cluster bootstrap CIs; precision projection before freezing n; no p-value hunting — one primary rule. |
| External validity | Single dataset, two scan types reported separately; claims limited to pelvic/femur CT; transfer stated as future work. |
| Reproducibility | All inputs hashed (`validation_inference_manifest.json`, config SHA-256); patch on ABBC `ccc625f`; outputs identified by hash since inference is not bit-reproducible. |

## Limitations (By Design)

- **Hypothesis generated on the validation set.** Mitigation: held-out confirmation; the paper states the sequence explicitly.
- **n = 100 held-out cases.** Mitigation: precision projection in WP1; fallback framing pre-declared.
- **Partition risk labels are coarse** (IoU ≥ 0.1; E5). Mitigation: pre-registered volume-fraction secondary risk; limitation stated.
- **M = 3 members.** No ensemble-size ablation in this paper.
- **Backend specifics** (ABBC two-stage; 159 pelvic splits in Stage 2). Mitigation: scores are backend-agnostic by construction; stated.
- **One primary normalisation.** Alternatives (e.g. `VI / max(H(P),H(Q))`, `VI / log k`) listed as untested.

## Ethical Considerations

- Public, de-identified challenge data; comply with PENGWIN licence, citation and publication rules (the embargo question in `stage0_readiness.md` must be closed before submission).
- No human subjects, no new data collection → human-subjects administrative section not applicable.
- AI-assisted research tooling disclosed in the paper's acknowledgements per EMBC/IEEE policy.
- Dual-use: none identified.

## Reporting Standard

- Closest fit: **CLAIM** (Checklist for Artificial Intelligence in Medical Imaging) for the model/evaluation description, trimmed to the 4-page limit; TRIPOD+AI items on data splits and pre-specification where applicable. Exact checklist versions to be verified in Phase 2.

## Preregistration

- Recommended: **Yes** (confirmatory comparison).
- Platform: in-repository frozen config (`configs/stage3_h1_v2.json`, committed, SHA-256 cited); optional OSF registration of the same document if the authors want an external timestamp.
- Status: Planned (WP1).
- Completed artifact declaration: `not_provided`.
- Companion handle: none.
- Sidecar ownership: dispatching layer only; no digest populated here.
