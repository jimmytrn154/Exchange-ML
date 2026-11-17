# Research Question Brief

Phase 1 · `research_question_agent` · 2026-11-17 · inputs: `plan.md`, `__docs__/paper.md` §3, `__docs__/reports/stage3/*`, `outputs/stage3_h1/*`.

## Topic Area

Failure detection for 3D pelvic-fracture instance segmentation using ensemble disagreement over predicted instance maps. Two complete H1 runs on PENGWIN showed that the proposed score (FA-IPD v1) collapsed onto foreground-presence disagreement and gave no advantage over voxel entropy, while a plain partition term (union-NVI) carried the only merge/split signal. The topic is what that disagreement actually measures and how it should be evaluated.

## Primary Research Question

> In 3D pelvic-fracture instance segmentation, does the **partition** component of ensemble instance-map disagreement, normalised to a bounded scale, detect merge, split and missing-fragment failures better than the **presence** component and than voxel-wise predictive entropy?

(Single comparative question. The "evaluation conditions" theme is Sub-question 3, not a second clause.)

## FINER Assessment

| Criterion | Score | Justification |
|---|---|---|
| Feasible | 4/5 | Data (340 labelled PENGWIN cases), three trained ensemble members, per-bone metric code (`instances.py` `per_anatomy`) and the scoring/evaluation CLIs exist. Only new compute is 300 held-out inferences (≈ 8–56 GPU-h). Risk: server instability (3 reboots in Oct) and a 1-week schedule slack. |
| Interesting | 4/5 | Genuine puzzle: a structural score with a sound definition behaved as a foreground score. The explanation (normalisation scale) is simple, checkable and has consequences for anyone who adds a VI/NVI term to a voxel-scale score. |
| Novel | 3/5 | NVI normalisation variants are well studied in clustering (Meilă; Vinh, Epps & Bailey 2010 — to be verified in Phase 2). Structural/topological uncertainty for segmentation QC exists (Gupta 2023, Yang 2025 — to be verified). What is not established, to current knowledge, is (a) the presence/partition decomposition of instance-map disagreement, (b) the scale failure at voxel counts of 10⁵–10⁶, (c) the evaluation conditions C3 quantified on a fracture-instance task. Prior-art check is Phase 2's job; score may move ±1. |
| Ethical | 5/5 | Public, de-identified challenge data under its licence; no human subjects, no new data collection. Data-use and challenge-rule compliance statement required (see Stage 0 report). |
| Relevant | 4/5 | Fragment count/configuration drives pelvic-fracture surgical planning (clinical claim to be sourced in Phase 2). A GT-free flag for merge/split errors is a direct QC tool. EMBC audience is broad biomedical engineering, so the clinical hook must be explicit. |
| **Average** | **4.0/5** | Threshold met (≥ 3.0, no criterion < 2). |

## Scope Boundaries

**In scope**
- PENGWIN 2026 Task 1 labelled data: 40-case validation split (diagnosis set) and a pre-registered 100-case held-out subset of the AL pool (confirmation set).
- The frozen three-member ensemble (seeds 31001–31003) and the Stage 2 single model (seed 31000) as the secondary risk source.
- Scores: mean/p95 entropy, pairwise Dice disagreement, foreground Jaccard disagreement, union-NVI, fragment-count variance, Hungarian disagreement, FA-IPD v1 (frozen), FA-IPD v2 (pre-registered), and the separated presence and partition terms.
- Risks: per-bone `1 − instance F1`, normalised merge+split count, missing-fragment recall, `1 − Dice` as semantic control.
- Evaluation conditions C3(a)–(d) quantified on our data.
- H4 controlled perturbations (merge / split / missing / boundary) as mechanism check.

**Out of scope**
- Active learning, acquisition functions, label-efficiency curves (H2) — deferred; vocabulary banned from the paper.
- RibFrac or any second dataset (H3) — deferred.
- The 50-case PENGWIN test split and the hidden challenge test set — sealed.
- New segmentation architectures, training-recipe changes, or any change to the v1 protocol.
- Alternative NVI normalisations beyond the one pre-registered primary (listed only as a limitation).

**Domain:** medical image analysis / segmentation quality control.
**Timeframe:** data 2026 PENGWIN release; experiments Nov 2026 – Jan 2027; literature through Nov 2026.
**Geography:** not applicable (public multi-centre dataset).
**Population:** 3D CT volumes with pelvic (sacrum, left hip, right hip) or femur fractures; unit of analysis = (case, annotated bone).

**Key assumptions**
1. The frozen member checkpoints are representative segmenters for this task (Stage 2 instance F1 0.850 after fixes).
2. Per-bone GT instance labels in PENGWIN are complete for the bones the scan type annotates (D4: other visible bones are unlabelled by design, hence the per-bone unit).
3. The 100 held-out pool cases were never seen by any member (true by construction of the Stage 0 split and trajectory-1 initial set).
4. Merge/split counts under IoU ≥ 0.1 matching are an acceptable, if coarse, structural risk (E5 recorded as a limitation).

## Sub-questions

1. **(C1, in hand)** How does pairwise instance-map disagreement split between its presence and partition terms on the 40 validation cases, and how does that split change when the partition term is normalised by `log N_voxels` (v1) versus by the joint entropy `H(P,Q)` (v2)?
2. **(C2, held-out)** On pre-registered held-out units, is the merge+split AURC of the v2 partition score lower than that of foreground Jaccard disagreement and of mean entropy, with bootstrap CIs excluding zero?
3. **(C3, in hand + held-out)** Which of the four evaluation conditions — risk-source model mismatch, binary-failure prevalence saturation, annotation-scope confound, IoU-threshold blindness to partial loss — change the ranking of scores, and by how much?

### Sub-Question Bindings

1. inherits: population = (case, bone) units of the 40 validation cases; domain = segmentation QC; timeframe = existing Oct-2026 member outputs; deviations: none.
2. inherits: population = (case, bone) units of the 100 held-out pool cases; domain, timeframe as above; deviations: none.
3. inherits: both populations; deviations: none. Condition (a) additionally uses Stage 2 single-model risk as the mismatched source — within the in-scope list.

## Candidate Questions Considered

| # | Candidate | FINER avg | Why not selected |
|---|---|---|---|
| 1 | Does the bounded-scale partition component detect merge/split/missing failures better than presence and entropy? | 4.0 | **Selected.** Comparative, implies the pre-registered design directly. |
| 2 | Does FA-IPD v1 detect structural failures better than entropy? | 2.6 | Already answered twice (no); Novel 1/5. Reported as background, not a question. |
| 3 | Does ensemble disagreement predict instance F1 in fracture segmentation? | 3.0 | Too broad; presence term dominates, so the answer would restate C1 without isolating the mechanism. |
| 4 | Does the choice of NVI normalisation change failure-detection rankings? | 3.4 | Real but narrow; absorbed as Sub-question 1. |
| 5 | How do partial-annotation and prevalence confounds distort uncertainty evaluation? | 3.4 | Methodological; absorbed as Sub-question 3. |
