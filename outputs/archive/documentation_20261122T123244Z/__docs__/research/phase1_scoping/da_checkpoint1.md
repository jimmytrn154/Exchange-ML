# Devil's Advocate Report — Checkpoint 1

Phase 1 · `devils_advocate_agent` · 2026-11-17 · reviews `rq_brief.md` + `methodology_blueprint.md`.

## Verdict: PASS (with three Major issues to fold into the blueprint before freeze)

## Critical Issues (Blocks Progression)

No critical issues identified.

## Major Issues

1. **The "a priori" normalisation was chosen after seeing it fail.**
   - **Type:** Method / moving goalposts risk.
   - **Location:** Blueprint §B; `plan.md` V1.
   - **Problem:** `VI / H(P,Q)` is called a priori, but it was selected after `VI / log N` collapsed on the validation set. A reviewer will read "pre-registered" and then find the hypothesis was reverse-engineered from the same 40 cases that are also reported in Table 1.
   - **Recommendation:** Keep the design, change the language. The paper must say: v1 failed on validation → diagnosis → v2 specified and frozen → tested on held-out cases. Never call v2 "a priori"; call it "specified before held-out scoring". Report validation and held-out columns side by side and let the reader see the sequence.

2. **The held-out sample size is set by the GPU budget, not by the effect.**
   - **Type:** Evidence / statistical power.
   - **Location:** Blueprint, Data Strategy.
   - **Problem:** The validation effect for the primary endpoint is small (merge+split AURC 0.739 vs 0.798, Δ ≈ 0.06; union-NVI vs FA-IPD v1, which is only a proxy for v2 vs Jaccard). With 100 cases of which only 50 are pelvic (the only scans that contribute three bones and most merges/splits), a null CI is plausible. The blueprint's precision projection is the right fix but is currently a promise.
   - **Recommendation:** Make the projection a WP1 deliverable with a number in it before the config is committed. Pre-declare the rule: if the projected half-width > Δ_validation, either raise n (all 210 pool cases; ≈ 17–117 GPU-h) or declare the analysis-paper framing in advance. Do not discover this in January.

3. **Member-mean risk shares construction with the score; condition (a) must be shown both ways.**
   - **Type:** Method / circularity.
   - **Location:** Blueprint, risk sources.
   - **Problem:** With the same three members producing both the disagreement and the averaged error, *any* disagreement term is mechanically related to error (a member that misses a bone raises both Jaccard disagreement and mean risk; a member that splits raises both NVI and mean split error). That is the intended construct for failure detection, but it means the comparison between presence and partition is fair only because both sit under the same construction. The Stage 2 single-model risk removes the mechanical link but reintroduces E1.
   - **Recommendation:** Report both risk sources for every score row, not just for condition (a). State the circularity explicitly: "the score and primary risk are computed from the same ensemble; the secondary risk uses an independent model." If the C2 conclusion holds under both, it is robust; if only under the primary, say so.

## Minor Issues

- The missing-fragment risk (recall < 1 under IoU ≥ 0.1) will be near-saturated for pelvic bones with many small fragments; expect prevalence problems like E2. Pre-specify the top-quartile rule for it too.
- "Fragment count" as a confounder is itself an output of the models (mean member count) in GT-free scoring but a GT quantity in evaluation. State which is used where (evaluation: GT count).
- 200 units from 100 cases, but pelvic cases carry 3 units and femur cases 1: the bootstrap must resample cases, which the blueprint says; also report per-scan-type AURC so a pelvic-only effect is visible.
- The analytical bound `VI ≤ 2 log k` assumes both partitions have ≤ k parts; state k as the max over the pair.

## Observations

- Union-NVI (v1-normalised, whole-ROI) already beat everything on merge+split AURC on validation. If v2 only matches union-NVI on held-out, the paper's contribution shifts toward C1 + C3 (decomposition and evaluation conditions) with "use the partition term alone" as the practical recommendation. That is still a paper; the framing table in `plan.md` §3.3 should name this middle outcome explicitly.
- H4 perturbation curves are the cheapest, most reviewer-proof figure in the paper. Prioritise them in WP3 even if GPU time slips.

## Strongest Counter-Argument

"This is a negative result dressed as a method paper. FA-IPD did not work; the authors re-normalised the term that happened to work on their validation set, tested it on 100 more cases from the same dataset and the same models, and the effect is a 0.06 change in AURC under a risk label (merge/split counts at IoU ≥ 0.1) that they themselves show is blind to partial losses. The 'evaluation conditions' are post-hoc explanations for why their first result was null."

Response the paper must be able to give: the sequence is stated; the held-out test is a genuine prediction with a decision rule fixed in advance; the evaluation conditions are quantified, not narrated; and the decomposition result (C1) is a positive, general finding independent of whether C2 passes.

## What's Missing

- A clinical citation for "fragment configuration matters for surgical planning" — the Relevance score depends on it (Phase 2 task).
- Prior-art check on presence/partition decomposition of instance disagreement and on NVI normalisation at voxel scale (Phase 2 task; Novelty score provisional).
- A statement of what counts as a "meaningful" AURC difference beyond CI-excludes-zero (e.g. ≥ 0.02 absolute). Suggest adding a minimal effect size to the decision rule before freeze.

## Stress Test Results

| Test | Result |
|---|---|
| Remove strongest source (held-out C2 result) — does the argument hold? | Yes: C1 (decomposition + scale bound) and C3 (conditions) stand on validation data alone. |
| Flip the research question — is the opposing view credible? | Partly: "presence disagreement is all you need for QC" is credible for missing-bone errors; the paper should concede this and position partition as complementary for merge/split. |
| Apply to a different context — does the finding generalise? | Unknown; single dataset. The scale bound (C1) generalises analytically; the empirical effect does not yet. |
| "So what?" — is the significance justified? | Yes for practitioners adding VI-type terms to voxel-scale scores; conditional for clinical QC pending C2. |
