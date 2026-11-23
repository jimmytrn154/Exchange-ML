# Stage 2.5 Integrity Report — manuscript overhaul (pre-review)

Date: 2026-11-23. Input: `paper/manuscript.tex` SHA-256 recorded in `pipeline_state.json` (f0cd51d9…). Mode: pre-review (mid-entry, existing draft). No experiment was re-run; every check reads saved artifacts.

## Phase A — references (10/10 registered)

| Key | Status | Evidence |
|---|---|---|
| lakshminarayanan2017 | VERIFIED | NeurIPS 30 (2017) |
| vinh2010 | VERIFIED | JMLR 11:2837–2854 |
| isensee2021 | VERIFIED | Nat. Methods 18:203–211, doi:10.1038/s41592-020-01008-z |
| reinke2024 | VERIFIED | local Crossref record `review/sources/crossref_s41592-023-02150-0.json` |
| maier2024 | VERIFIED | local Crossref record `review/sources/crossref_s41592-023-02151-z.json` |
| valindria2017 | VERIFIED | IEEE TMI 36(8):1597–1606 |
| schilling2023 | VERIFIED | local full text `review/sources/schilling2023.pdf` |
| zenk2025 | VERIFIED | MedIA 101:103392, doi:10.1016/j.media.2024.103392 (arXiv 2406.03323) |
| abbc | VERIFIED | matches `.gitmodules` URL and pinned commit ccc625f |
| pengwin2026 | **UNCONFIRMED (author check)** | Web search did not surface Zenodo record 19732767; the 2026 challenge site exists (pengwin2026.grand-challenge.org). Author should confirm DOI/record against the download source. |

## Phase C — reported data (all numeric statements traced)

Traced to `outputs/stage3_h1_v2/{tables/decomposition.csv, perturbations/summary.json, heldout/comparison.csv, heldout/paired_differences.csv, heldout/risks.csv, heldout/scores.csv}`, `outputs/evaluator_sensitivity_v1/paired_differences.csv`, `configs/stage3_h1_v2.json`.

- Median weighted partition 0.001164 → 0.032116; share 6.958% → 58.871%; VI ≤1e-12 in 27/80 (0.3375): MATCH.
- 32 phantom checks; 527 responses / 452 eligible; 240 contingencies: MATCH.
- AURCs (Table I), ΔAURC −0.16313 [−0.21184, −0.11682], −0.30470 [−0.37159, −0.23683]; Stage-2 −0.12760, −0.28142: MATCH.
- FA-v2 − partition_v1 +0.05914 [0.03921, 0.08032] (E3 bootstrap) and +0.0591 [0.0393, 0.0801] (sensitivity bootstrap, Table II): both MATCH their sources; two independent bootstraps of the same contrast.
- Femur-only CIs cross zero: MATCH. Case 187/left_hip n_GT=5, all scores 0, missing volume 1, M+S 0: MATCH.
- 210 candidates (105+105), 10,000 resamples, criterion −0.02, prune 1000 mm³, IoU 0.1, 26-connectivity: MATCH config.

Unused but available evidence relevant to the argument (no new analysis needed): random-ranking AURC 0.6009 (M+S) and 0.1964 (1−F1) in `comparison.csv`; D_joint − D_vox = −0.11605 [−0.15693, −0.07707]; Spearman ρ per score.

## AI Research Failure-Mode Checklist

| Mode | Outcome | Evidence |
|---|---|---|
| 1 Implementation bug | CLEAR | Hash-verified manifests (111 output + 142 cache), 271 unit tests, 32 phantom assertions, count-replication residuals 0 |
| 2 Hallucinated citation | CLEAR with 1 author check | pengwin2026 record unconfirmed online |
| 3 Hallucinated result | CLEAR | All numbers traced above |
| 4 Shortcut reliance | FLAG (advisory) | Pooled result driven by pelvic bones (femur CIs cross 0; femur M+S prevalence 96%); must be disclosed |
| 5 Bug as insight | CLEAR | Entropy below random is consistent with structural errors under confident voxels; no "surprising" framing used |
| 6 Methodology fabrication | CLEAR | Methods values match config/code |
| 7 Frame-lock | **SUSPECTED** | Draft is framed around the combined score D_joint, while its own evidence shows the partition component carries the structural signal (partition-only best) and that components are complementary. The contribution is currently explained *despite* the framing. Remedy: return to WRITE (reframing) — matches the author's overhaul request. |

## Other integrity notes
- Acknowledgment discloses only OpenAI Codex; the current revision is AI-assisted by Anthropic Claude. IEEE disclosure should name all tools used (author decision on wording).
- "Recorded comparison rule, not formal preregistration": config records the normalization as "chosen after E1 diagnosis and specified before E3"; chronology is not independently timestamp-verified. The manuscript may state the development→fixed→held-out design but must not claim preregistration.

## Verdict
Data and references: PASS (one author confirmation pending). Failure-mode gate: BLOCK on Mode 7 pending user acknowledgement → recommended action: confirm flag and proceed to rewriting.

---

# Stage 3 / 4 / 4.5 — overhaul revision and final integrity (2026-11-23)

Author decisions recorded at the Stage 2.5 checkpoint: (1) Mode 7 flag confirmed → reframe around coverage vs grouping, keeping the recorded primary criterion (D_joint vs coverage/entropy) and reporting grouping-only superiority as secondary; (2) Fig. 1 reuses the four real CT/prediction panels of the provenance-tracked `figures/overview.png`; (3) add ≤8 web-verified references; (4) disclose both AI tools.

**Stage 3 reviewer audit → Stage 4 fixes.** Causal wording on confident voxels softened; "significantly" replaced by the effect size; design rationale for the combined primary score stated; "explain" → "suggest"; missing volume listed among alternative definitions; Spearman named at first use; abstract closing claim tempered.

**Stage 4.5 fresh pass.**
- Phase A: 18/18 references; 8 added entries verified by web search (Sang 2026 TMI, Meilă 2007, Geifman 2017, Jaeger 2023, Roy 2019, Kirillov 2019, Mehrtash 2020, Siddiqui 2024). `pengwin2026` Zenodo record still needs author confirmation.
- Phase C: `review/check_manuscript_numbers.py` recomputes every quantitative statement from saved CSV/JSON evidence → **58/58 PASS**. New derived numbers (risk at 30% coverage, omission ranks, replication ratio, merge medians, log|C|) are written by `make_figures.py` into `evidence/figures.json`; risk–coverage curves are recomputed from saved scores and must reproduce saved curves/AURCs within 1e-9.
- Failure modes: 1, 3, 5, 6 CLEAR (traced numbers, no new experiment); 2 CLEAR except pengwin2026 author check; 4 disclosed (femur intervals include zero; gains concentrated in pelvic bones); 7 RESOLVED by reframing with the primary criterion unchanged.
- Layout: local pdflatex (TinyTeX) → 5 pages; technical text ends on page 4; page 5 holds acknowledgment and references only. Tectonic (the project's canonical compiler) was unavailable on this machine.

Verdict: PASS with one open author action (pengwin2026 DOI).
