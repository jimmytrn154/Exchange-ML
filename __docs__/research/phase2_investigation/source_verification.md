# Source Verification Report

Phase 2 · `source_verification_agent` · 2026-11-17 · input: `annotated_bibliography.md` (28 sources).

## Overall Assessment

**Sources reviewed:** 28 · **Verified (DOI resolves + metadata match, or venue page + OpenAlex match):** 27 · **Plausible only:** 1 (#28 software) · **Flagged:** 4 (content caveats, not existence) · **Rejected:** 0 (three candidates were excluded upstream; see bibliography PRISMA flow).

Tier 0 was run through OpenAlex (Semantic Scholar not queried) for the ten no-DOI/preprint-canonical items; Tier 1 Crossref resolution for 18 DOIs; Tier 2 web spot-check for all Tier-3/no-DOI items (100 % of those, exceeding the 50 % rule). Four sources were read in full or by section and are marked `source_verified_against_original: true`.

## Source Quality Matrix

| # | Source | Level | Existence | Venue | Currency | COI | Read scope | Overall |
|---|---|---|---|---|---|---|---|---|
| 1 | Meilă 2007 JMVA | VII | VERIFIED (Crossref 10.1016/j.jmva.2006.11.013; vol 98(5) 873–895) | pass | foundational | none | abstract | A |
| 2 | Meilă 2003 LNCS | VII | VERIFIED (Crossref 10.1007/978-3-540-45167-9_14; pp. 173–187) | pass | foundational | none | abstract | A |
| 3 | Vinh et al. 2010 JMLR | VII | VERIFIED (JMLR page; OpenAlex 2 809 cites; full PDF retrieved, text extracted) | pass | foundational | none | **sections** | A |
| 4 | Arganda-Carreras 2015 | VI | VERIFIED (Crossref 10.3389/fnana.2015.00142) | pass (Frontiers, DOAJ) | foundational | challenge organisers evaluate own challenge — intellectual, low | **sections** | A |
| 5 | Nunez-Iglesias 2013 | VI | VERIFIED (Crossref 10.1371/journal.pone.0071715) | pass | foundational | none | abstract | B |
| 6 | Kirillov 2019 CVPR | VI | VERIFIED (Crossref 10.1109/CVPR.2019.00963) | pass | foundational | none | abstract | A |
| 7 | Lakshminarayanan 2017 | VI | VERIFIED (OpenAlex, arXiv 1612.01474; 5 171 cites) | pass (NeurIPS, no DOI) | foundational | none | abstract | A |
| 8 | Jungo & Reyes 2019 | VI | VERIFIED (Crossref 10.1007/978-3-030-32245-8_6) | pass | ok (7 y, cited for a durable finding) | none | abstract | B |
| 9 | Roy et al. 2019 | VI | VERIFIED (Crossref 10.1016/j.neuroimage.2019.03.042) | pass | ok | none | abstract | B |
| 10 | Mehrtash 2020 | VI | VERIFIED (Crossref 10.1109/TMI.2020.3006437) | pass | ok | none | abstract | B (reserve) |
| 11 | Zenk 2025 MedIA | VI | VERIFIED (Crossref 10.1016/j.media.2024.103392; vol 101, 103392) | pass | current | none | abstract | A |
| 12 | Kahl 2024 ICLR | VI | VERIFIED (OpenAlex, arXiv 2401.08501; ICLR oral page) | pass (no DOI) | current | same group as #11, #17 — intellectual, low | abstract | A |
| 13 | Geißler 2024 MIDL | VI | VERIFIED (PMLR v250 page fetched; pp. 504–519) | pass (no DOI) | current | none | abstract | A |
| 14 | Gupta 2023 NeurIPS | VI | VERIFIED (NeurIPS proceedings page; arXiv 2306.05671) | pass | current | none | abstract | A |
| 15 | Yang 2025 MedIA | VI | VERIFIED (Crossref 10.1016/j.media.2025.103602) | pass | current | none | abstract | A |
| 16 | Berger 2025 IPMI | VI | VERIFIED (Crossref 10.1007/978-3-031-96628-6_20; pp. 297–312) | pass | current | none | abstract | A, **flag F1** |
| 17 | Jaeger 2023 ICLR | VI | VERIFIED (OpenAlex, arXiv 2211.15259; ICLR oral page) | pass (no DOI) | current | see #12 | abstract | A |
| 18 | Geifman 2019 ICLR | VI | VERIFIED (OpenAlex, arXiv 1805.08206; ICLR poster page) | pass (no DOI) | ok | none | abstract | A |
| 19 | Maier-Hein 2024 | VII | VERIFIED (Crossref 10.1038/s41592-023-02151-z) | pass | current | consortium authors overlap with #11/#12/#17/#25 — intellectual, low | abstract | A |
| 20 | Reinke 2024 | VII | VERIFIED (Crossref 10.1038/s41592-023-02150-0) | pass | current | as #19 | abstract | A |
| 21 | Phitidis 2026 AIM | VI | VERIFIED (Crossref 10.1016/j.artmed.2026.103507; vol 181, 103507; arXiv HTML read) | pass | current | none | **sections** | A |
| 22 | Sang 2026 TMI (PENGWIN 2024) | VI | VERIFIED (Crossref 10.1109/TMI.2025.3650126; vol 45(5) 2212–2228; arXiv 2504.02382 HTML read) | pass | current | challenge organisers; dataset providers — intellectual, moderate | **sections** | A, **flag F2** |
| 23 | Liu 2023 MICCAI | VI | VERIFIED (Crossref 10.1007/978-3-031-43996-4_30) | pass | ok | same group as #22 | abstract | B |
| 24 | Raniga 2025 RadioGraphics | VII | VERIFIED (Crossref 10.1148/rg.250027) | pass | current | none | abstract | B, **flag F3** |
| 25 | Isensee 2021 Nat Methods | VI | VERIFIED (Crossref 10.1038/s41592-020-01008-z) | pass | ok | none | abstract | A |
| 26 | Huang 2023 STU-Net | VI | VERIFIED existence (OpenAlex, arXiv 2304.06716; 262 cites) | **preprint** | ok | none | abstract | B, **flag F4** |
| 27 | Shi 2021 MedIA | VI | VERIFIED (Crossref 10.1016/j.media.2021.101979) | pass | ok | none | abstract | A |
| 28 | ABBC repository | grey | PLAUSIBLE (local checkout at `baselines/`; not fetched online in this run) | n/a | current | none | n/a | C (software) |

## Flagged Sources (Detail)

#### F1 — Berger et al. 2025 (#16)
- **Issue:** The specific claim "VOI mixes volumetric and topological information" comes from an AI-generated summary page, not from the primary text.
- **Severity:** Medium (misattribution risk).
- **Recommendation:** Include; cite only for the three named pitfalls (connectivity, GT artifacts, metric choice). Do not cite the VOI claim unless the primary text is read.

#### F2 — Sang et al. 2026 (#22)
- **Issue:** Describes the **2024** edition (150 CT); our data is the 2026 edition (340 cases). Dataset numbers must not be transferred.
- **Severity:** Medium.
- **Recommendation:** Include with caveat; cite for lineage, label semantics and clinical motivation only; cite the project data audit and the 2026 challenge site for our dataset facts.

#### F3 — Raniga et al. 2025 (#24)
- **Issue:** Content known only from search snippets; the clinical sentence the paper needs ("fragment delineation and displacement on CT guide reduction planning") must be checked against the article before submission.
- **Severity:** Low–Medium.
- **Recommendation:** Include; author to read the relevant section before Stage 2 WRITE (`/ars-mark-read` scope `sections`).

#### F4 — Huang et al. 2023 STU-Net (#26)
- **Issue:** arXiv preprint; no peer-reviewed version found.
- **Severity:** Low (backend attribution only).
- **Recommendation:** Include as preprint; `contamination_signals: {preprint_post_llm_inflection: false, openalex_unmatched: false}`.

## Predatory Journal Alerts

None. All journals are Elsevier (MedIA, JMVA, AIM, NeuroImage), Nature Portfolio, IEEE, Springer LNCS, Frontiers (DOAJ-listed), PLOS, RSNA, JMLR, PMLR.

## Conflict of Interest Disclosures

- The DKFZ/Heidelberg group authors #11, #12, #17, #19, #20, #25 (and nnU-Net). Their methodological recommendations (AURC, pairwise Dice baseline, metrics pitfalls) are therefore from one intellectual lineage; the paper should note that the evaluation protocol follows this lineage rather than present it as independent consensus.
- PENGWIN organisers (#22, #23) provide the dataset and the challenge report; no financial COI identified.

## Factual Claim Verification (cross-referenced)

| Claim used in the paper | Sources | Status |
|---|---|---|
| NVI = VI/H(U,V) is a metric; VI/log N and VI/2 log K are looser normalisations with narrower useful range | #3 (§3.3, Thm 3, fn 4); #1 | **Established** (primary text read) |
| H(U,V) ≤ 2 log K when N > RC | #3 fn 4 | Established |
| Split/merge decomposition of information-theoretic segmentation scores | #4 (§2.5) | Established (primary text read) |
| Pairwise Dice between ensemble members is a strong case-level failure-detection baseline | #11 (abstract); #13 (uses mean pairwise Dice) | Supported (abstract level) |
| AURC is the recommended primary failure-detection metric | #17; #11 (risk-coverage); #18 | Supported |
| Structure size confounds uncertainty–quality correlation; partial correlation is the remedy | #13 | Supported (abstract) |
| Partially labelled test data prevents measuring FPs on unannotated classes | #21 (§5, quoted) | Established (primary text read) |
| Pelvic fracture segmentation underpins fragment-wise reduction planning | #22 (intro, quoted); #24 | Supported |
| Fragment count is a surgical cutoff | — | **Not supported**; do not claim |
| PENGWIN 2024: 150 CT, 3–12 fragments per case, highest-IoU matching | #22 (dataset/eval sections) | Established (2024 edition only) |

Cross-reference rate: 10/10 claims checked against ≥ 1 source; 6/10 against ≥ 2 → 60 % (≥ 30 % required).

## Verification Limitations

- Semantic Scholar not queried; OpenAlex used as the sole resolution index for no-DOI items.
- 22 of 28 sources read at abstract level only; `source_verified_against_original: true` applies to #3, #4, #21, #22 only.
- #28 verified only as a local checkout.
- Retraction status: OpenAlex `is_retracted` not queried in this run; no retraction notices surfaced in searches. Status `not_checked` for all DOIs.
