# EMBC 2027: venue evidence and review scope

Checked 22 November 2026. This is a targeted literature comparison, not a systematic review or an estimate of acceptance probability. Accepted papers show examples of contributions the venue has published; they do not identify a minimum threshold, the rejected-paper distribution, or the 2027 reviewer decisions.

## Verified official requirements

- [2027 author guidelines](https://embc.embs.org/2027/author-guidelines/): four technical pages plus a fifth page for acknowledgments/references, at most 20 references; original English work; AI-generated contributions require disclosure; a qualified reciprocal reviewer or valid exemption must be declared. The biomedical-imaging theme fits the study. The page does not prescribe a minimum patient count, dataset count, particular architecture, improvement percentage, external cohort or clinical trial for every paper.
- [2027 call for papers](https://embc.embs.org/2027/papers/): full-paper deadline 24 January 2027, 23:59 AoE; decision 16 April and final version 30 April. Translational relevance is emphasized, but the listed themes also include biomedical imaging and image processing.
- [2025 call for papers](https://embc.embs.org/2025/contributors/papers/): full papers then allowed 4–7 pages; review was single blind, assigned using themes/keywords, with qualitative and quantitative feedback considered. A six-page accepted 2023/2025 paper is therefore evidence about scientific scope, not a template for 2027 length. The public pages inspected do not provide a calibrated technical-score cutoff or an acceptance probability for this manuscript.

## Local materials inspected

`paper/guidelines.md` contains copied policy/CFP text; `paper/conference-latex-template.zip` contains the generic IEEE June-2024 example, class and documentation. The example `.tex` describes formatting rather than this study. `paper/source/` contains no manuscript. Scientific assessment uses the existing Results/Discussion working draft and audited E1–E5 artifacts, not an imagined finished paper.

## Selection and extraction rules

Include EMBC full conference papers closely related to segmentation uncertainty, quality/failure assessment or instance review. Verify conference/year/DOI against proceedings, PubMed or institutional author records. Extract experimental details only from accessible full text; identify abstracts and accepted-program-only records separately. Distinguish patients, images, slices and bone units. Do not compare Dice/AURC values across unrelated tasks as a league table. Add a small set of non-EMBC primary studies solely to test novelty. Missing detail in an accessible version is not proof that an experiment was never conducted.

Searches used combinations of `EMBC`, `segmentation`, `uncertainty`, `quality assessment`, `failure detection`, `ensemble`, `Bayesian`, `review`, and publication years. This is a relevance-selected sample; no exhaustive database hit count or PRISMA flow is claimed. Exclude unrelated EEG/device papers and venue-unverified preprints from the accepted full-paper comparison. Do not import acceptance-rate estimates from commercial conference-summary sites.

## Current local evidence audit

Read-only `python -B scripts/audit_e1_e5_evidence.py` passed again: 111 recorded output hashes and 142 recorded cache hashes. This validates the recorded evidence chain, not independent re-inference of every volume. On 22 November, the E6 external root has complete preparation/preprocessing manifests and two member training manifests marked `running`; no completed E6 external evaluation manifest was found. No E6 result is used to support the present scientific assessment.

No training, scoring, inference or bootstrap was rerun for this review. No manuscript, frozen config or experiment output was modified.
