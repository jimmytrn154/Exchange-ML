# Exchange-ML: local evidence and novelty audit for EMBC 2027

**Date:** 22 November 2026. **Scope:** read-only scientific assessment supporting the requested focused literature review. This is neither an acceptance prediction nor a complete manuscript integrity clearance. Existing experiment outputs were not changed or recomputed.

## 1. What is actually in `paper/`

- `paper/source/` is empty. There is no complete manuscript, bibliography or study-specific LaTeX source there.
- `paper/guidelines.md` contains a policy summary followed by copied EMBC 2027 call-for-papers material. It is a submission-policy resource, not research content.
- `paper/conference-latex-template.zip` contains the generic June 2024 IEEE conference template: sample `.tex`, matching example PDF, `IEEEtran.cls`, `IEEEtran_HOWTO.pdf`, example figure and synchronization file. The complete sample `.tex` was read; its example references and figures are placeholders, not Exchange-ML material. The `.cls` and HOWTO are typesetting support, not evidence of study quality.
- The relevant scientific writing is currently `__docs__/reports/results_discussion_draft.md`; line 3 explicitly says it is a Results/Discussion draft, not a complete submission-ready paper. Its internal writing notes at lines 80–86 also still require integration with Method, bibliography, format and protocol chronology.

An attempted optional local PDF text extraction could not run because `pdftotext` is not installed. The ZIP inventory and complete matching `.tex` establish the template's identity; no claim is made that every PDF instruction page was independently parsed.

### Policy cross-check

The [official EMBC 2027 author guidelines](https://embc.embs.org/2027/author-guidelines/) were opened on 22 November 2026. The local summary agrees on four technical pages plus a fifth page for acknowledgments/references, at most 20 references, original unpublished work, English, and a maximum of 35 authors. The official page also gives AI disclosure details and reciprocal-reviewer qualifications/exemptions. These are compliance conditions, not scientific acceptance thresholds. The complete manuscript cannot yet be checked against them because no complete manuscript exists in `paper/source/`. Local anchors: `paper/guidelines.md:1–20`, `paper/guidelines.md:35–50`, `paper/guidelines.md:80–92`.

## 2. Evidence independently refreshed

Agent-typed command:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/audit_e1_e5_evidence.py
```

Observed result: **PASS — 111 recorded output hashes, 142 recorded cache hashes, evidence sidecar verified.** This audit checks recorded report/cache/config/code lineage; it is not a new pixel-level evaluation or an independent replication. `outputs/stage3_h1_v2/heldout/summary.json` independently records 100 cases, 200 bone units, 400 inference records, 10,000 bootstrap draws and `smoke_only=false`. The E5 supplementary summary explicitly records `exploratory_post_heldout`.

## 3. What is supported, and where the claim must stop

| Evidence | Defensible interpretation | Material limit and local anchor |
|---|---|---|
| 40 validation cases/80 bones; E2 controlled interventions with 452 eligible responses and 32 phantom assertions | A controlled diagnosis of foreground-presence versus instance-partition disagreement, plus an exact denominator-mechanism check | Validation is diagnostic. Count replication is not CT resampling. `e1_e5_synthesis.md:9–16,24–60`. |
| FA-v2 pooled held-out M+S AURC 0.3311; presence 0.4942; mean entropy 0.6358 | Strong evidence for the configured comparison against those two baselines in this setup | Both member-mean contrasts have CIs below zero, but the result is not superiority over every control. `e1_e5_synthesis.md:62–94`; actual `heldout/summary.json`. |
| FA-v2 minus presence: −0.16313 [−0.21184,−0.11682]; minus entropy: −0.30470 [−0.37159,−0.23683] | Effect and uncertainty are reported, not just favorable point estimates | The scan-type strata are not equivalent confirmations: femur-only primary CIs cross zero. `e1_e5_synthesis.md:83–94`. |
| Partition-v1 AURC 0.2720; partition-v2 0.2978 | The partition signal itself is useful, and the decomposition reveals that the combined score is not optimal | FA-v2 is significantly worse than these component controls: Δ +0.05914 [0.03921,0.08032] and +0.03333 [0.01535,0.05265]. Do not promote a new winning primary after held-out inspection. `e1_e5_synthesis.md:78–94`. |
| Case-cluster bootstrap stratified by scan type | Properly carries dependent bones and repeated sampled cases | Pooled estimand still weights bones equally: 150 pelvic versus 50 femur units, despite 50/50 cases. This is a scope/estimand limitation, not proof of invalid CIs. `e1_e5_synthesis.md:64`; draft lines 11–13. |
| E4 normalization × scope comparison | Normalization, annotation attribution and aggregation need separate controls | Restricting annotation scope alone did not establish AURC improvement. Cross-unit E3/E4 AURCs cannot be compared as an isolated scope effect. `e1_e5_synthesis.md:98–122`. |
| FA-v2 top-severity-quartile enrichment 0.5829/0.5371 at review 10%/20% | Detecting any M+S differs from prioritizing the most severe risk | Those descriptive estimates are below the random reference 1; no significance claim was added. `e1_e5_synthesis.md:124–137`. |
| Fixed-budget M+S differences versus presence: −0.02653 [−0.05710,0.01200] and −0.00027 [−0.06011,0.05128] | Better whole-curve AURC does not guarantee a clear advantage at a practical operating budget | Both CIs cross zero. Missing-volume differences favor presence at both budgets; these are post-held-out, marginal exploratory intervals. `e1_e5_synthesis.md:145–159`. |
| Complete-miss example `187/left_hip`: all members miss the anatomy, disagreement=0, M+S risk=0 | Ensemble agreement cannot certify correctness; endpoint selection matters | No measured reader-time savings, correction success, clinical benefit or omission-safe QC policy has been established. `e1_e5_synthesis.md:159–163`; draft lines 76–78. |
| Signed and hashed configuration/output lineage | Useful traceability and preservation of corrected reporting | A hash does not establish independent preregistration. Timestamped pre-held-out commit/registration has not been verified; use the draft's bounded wording. `e1_e5_synthesis.md:18–22`; `results_discussion_draft.md:3`. |

These sources support an **analysis of when disagreement scores do and do not rank structural errors**, rather than a claim that FA-v2 is a new universal metric or the best practical review policy. This is a scientific framing assessment; the accepted-EMBC evidence collected separately is needed before making a venue-fit judgment.

## 4. Three concrete prior-work boundaries

The items below are deliberately outside the EMBC-only acceptance sample. They locate novelty; they must not be counted as EMBC precedents.

### Vinh, Epps and Bailey, JMLR 2010

[Publisher paper and full text](https://www.jmlr.org/papers/volume11/vinh10a/vinh10a.pdf): *Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance*, 11, 2837–2854. Theorem 3 and Table 3 (printed p.2842), followed by discussion on p.2843, already establish joint-entropy-normalized variation of information and discuss the comparatively loose `log N` bound. **Novelty boundary:** VI/H itself, its metric status on fixed partitions, and the generic normalization-range motivation are prior work. The possible contribution here is empirical diagnosis of how that known normalization behaves inside a foreground-weighted, pair-dependent 3D fracture-disagreement score. The local plan already acknowledges this source; see `plan.md:72–77,98`.

### Zenk et al., Medical Image Analysis 2025

[Published metadata](https://pubmed.ncbi.nlm.nih.gov/39657400/) and [institutional full text](https://inrepo02.dkfz.de/record/294922/files/1-s2.0-S1361841524003177-main.pdf), DOI **10.1016/j.media.2024.103392**: *Comparative benchmarking of failure detection methods in medical image segmentation: Unveiling the role of confidence aggregation*. Section 4.1 and Table 2 use risk–coverage/AURC on five public 3D datasets, with realistic shifts; Sections 5.2–6.2 compare aggregation and discuss strong pairwise-Dice controls and silent ensemble failures. **Novelty boundary:** neither using AURC, showing aggregation matters, nor observing confident ensemble errors is new by itself. The sharper distinction to establish is **instance-partition disagreement versus binary foreground disagreement for merge/split endpoints**, with all strong component controls retained. Its cross-dataset benchmark also explains why a broad generalization claim would need more than the existing single-backend PENGWIN result.

### Guarino et al., CVPR 2026

[Author full text](https://arxiv.org/html/2603.29941v1), *Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance*. The arXiv record states CVPR 2026 acceptance; a corresponding CVF proceedings PDF was returned by primary-site search, but direct CVF open returned HTTP403, so technical extraction here uses the author version. Sections 3.2–4 discuss aggregation pitfalls, spatial structure, task dependence and a ten-dataset benchmark. **Novelty boundary:** a broad claim that structural/spatial aggregation is neglected or first introduced here is unsafe. Exchange-ML would need to identify its narrower distinction: presence/partition decomposition over fracture-instance ensembles and the divergence between merge/split ranking, severe-error prioritization and omission-sensitive review. This paper is a current novelty comparator, not an EMBC acceptance-bar datum.

## 5. What E6 can change

**Current status refreshed by the root agent on 22 November:** the external output root has a complete 420-record preparation manifest and a complete preprocessing manifest; member01/member02 training manifests are running. There is no completed final E6 evaluation manifest or external-validation result yet. The 21 November implementation report's scheduler-blocked/full-unrun paragraphs are an older snapshot and must not be repeated as current pipeline status. Its method definitions remain useful: `e6_implementation.md:18–30,50`; protocol extension: `exps.md:268–286`.

- A completed E6 could test whether the **fixed score comparison** transfers to another dataset and backend/task setup, with held-out RibFrac cases and paired uncertainty estimates.
- It would not show that PENGWIN checkpoints transfer: E6 trains new binary nnU-Net models. Its fracture-lesion components are not complete rib/bone-fragment instances.
- It cannot automatically establish human-review savings, clinical utility or omission safety; those need the relevant outcome/cost evidence.
- If the new backend has too little variation in merge/split risk, that endpoint is uninformative, not a favorable result. All component controls and negative results must remain visible.
- Favorable results against presence/entropy still would not erase a stronger partition-only control. Conversely, an unfavorable E6 would narrow the generalization claim rather than invalidate every PENGWIN decomposition result.

## 6. Priorities supported by this audit

1. State the precise contribution as a controlled structural-failure analysis, and explicitly distinguish it from the three prior-work strands above. A new detector claim would require overcoming the stronger component controls and practical-budget limitations.
2. Complete E6 as specified and report effect/CI regardless of direction; keep source/task differences explicit. No change to the held-out PENGWIN method is proposed here.
3. Build the complete four-technical-page manuscript from the existing Results/Discussion, with a compact method figure, full relevant control table and one honest failure/limitation example. The existence of a template is not manuscript readiness.
4. Keep retrospective supplements labeled, reconcile protocol chronology wording, and apply the actual EMBC 2027 AI-disclosure and reviewer-registration rules before submission.

**User-typed commands prepared by this audit:** None. **Scientific experiment reruns:** None. **Files changed by this task:** this audit only. **Remaining question:** compare these evidence strengths and limitations with the separately verified EMBC-only literature sample; no numerical acceptance probability is supported.
