# Devil's Advocate Report — Checkpoint 2

Phase 3 · `devils_advocate_agent` · 2026-11-17 · reviews `synthesis.md` + `phase2_investigation/*`.

## Verdict: PASS (three Major issues; none blocks, all change what the paper may claim)

## Critical Issues (Blocks Progression)

No critical issues identified.

## Major Issues

1. **Alternative explanation for the 2 % partition share: the members may simply agree on partitions.**
   - **Type:** Evidence / alternative explanation.
   - **Location:** Synthesis Theme 1; `plan.md` C1.
   - **Problem:** The synthesis attributes the small partition term to the `log N` denominator. A rival explanation is that three seeds trained on the same 40 cases rarely disagree about how shared bone is split into fragments, so the numerator (VI) is near zero and no re-normalisation will create signal. The validation evidence that a partition signal exists (union-NVI AURC 0.739) is indirect, because union-NVI is computed on the **union** foreground and therefore mixes presence disagreement (foreground-vs-background label mismatches) into what it calls partition disagreement.
   - **Recommendation:** Two pre-registered checks in WP1 before the held-out run: (i) the distribution of the raw VI numerator and of v2 across the 40 validation units — if v2 has no spread, the rival explanation wins and the paper is C1 + C3 only; (ii) report union-NVI explicitly as a *mixed* variant and the v2 partition term (shared foreground only) as the *pure* variant. The H4 perturbation curves settle whether the pure term responds to merge/split at all.

2. **C1 is a re-expression, not a proposal — the paper must say "we show", not "we propose".**
   - **Type:** Novelty / scope.
   - **Location:** Synthesis Theme 1; `plan.md` §3.2 C1.
   - **Problem:** Foreground-restricted scoring and split/merge decomposition of information-theoretic scores are in Arganda-Carreras et al. (2015); the normalisation comparison and the `H(U,V) ≤ 2 log K` bound are in Vinh et al. (2010). The novel content is narrower: the numeric regime (N ≈ 10⁵–10⁶ voxels, K ≤ 12) and the empirical dominance of presence under `VI/log N` in an ensemble-disagreement setting.
   - **Recommendation:** Phrase C1 as an application of Vinh et al. to voxel-scale partition disagreement with new empirical evidence; cite Arganda-Carreras for the decomposition precedent. Drop any wording that implies the decomposition itself is new.

3. **Pre-registering NVI while the primary source prefers NID needs an explicit reason, or a reviewer will call it a convenient choice.**
   - **Type:** Method / selection rationale.
   - **Location:** Synthesis Contradictions row 2; Blueprint V1.
   - **Problem:** Vinh et al. state NID is "generally more preferable". The project picks NVI. The stated reasons (continuity with v1; VI = H(P|Q)+H(Q|P) splits into merge/split terms) are real but currently live only in the synthesis.
   - **Recommendation:** Put the reason in the v2 config and in the paper's method section; list NID as an untested alternative in limitations. Do not run NID on the validation set "just to see" — that would be exactly the post-hoc search the design forbids.

## Minor Issues

- The bounded-absence claim (Gap 1) relies on web search and two APIs; add one targeted OpenAlex query string for "instance" AND "ensemble" AND ("merge" OR "split") to the search log so the bound is reproducible.
- Theme 2 cites Mehrtash et al. for calibration at abstract level; it adds little and could be dropped to keep the paper under 20 references.
- Theme 5: Raniga et al. content is from snippets (verification flag F3); the author must read the relevant section before Stage 2.
- The ABBC repository has `<!--anchor:none:-->`; the finalizer will gate it. Acceptable for a software citation, but say so.

## Observations

- The synthesis is careful not to claim "first". Keep it that way; EMBC reviewers from outside the field react badly to priority claims they cannot check.
- The DKFZ-lineage caveat in Theme 4 is a strength if stated once; it shows the authors know whose protocol they are following.
- The honest outcome table in `plan.md` §3.3 now has a fourth row implied by Major 1: *v2 has no spread on validation* → the paper is the analysis paper and the held-out run is unnecessary for C2 (it can still serve C3). Add that row.

## Strongest Counter-Argument

"The literature already says overlap disagreement is the best simple failure detector (Zenk 2025), and the authors' own data agree. Their 'partition' term is a rescaled VI whose numerator is near zero because their ensemble members barely disagree on fragment layout. Rescaling a near-zero number does not make it informative. The one score that looked good — union-NVI — mixes presence back in. So the paper is an elaborate way of saying 'use pairwise Dice'."

The paper can answer this only with data the pipeline has not yet produced: the spread of the raw VI numerator on validation, the H4 perturbation response, and the held-out C2 result. If those come back empty, the counter-argument is correct and the honest paper is C1 + C3.

## What's Missing

- A per-unit distribution of the raw VI (unnormalised) on validation — the single cheapest check that decides Major 1.
- A sentence-level citation for the clinical hook (verification F3).
- A reproducible query string for the absence claim.

## Stress Test Results

| Test | Result |
|---|---|
| Remove strongest source (vinh2010) — does the synthesis hold? | Partly: C1's mechanism loses its primary citation; meila2007 alone gives the bounds but not the comparative statement. Keep vinh2010 central. |
| Flip the research question — is "partition disagreement is uninformative" credible? | Yes (Major 1). The design must be able to return that answer. |
| Apply to a different context — does the finding generalise? | The scale bound generalises to any voxel-scale VI normalisation; the empirical claims do not beyond PENGWIN. |
| "So what?" — is the significance justified? | Yes for anyone adding VI/NVI terms to segmentation uncertainty scores; conditional for clinical QC. |

```
[DA-DECISION: no rebuttals received at this checkpoint | ACTION: none | REASON: first emission]
```
