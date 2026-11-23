# First-reader review of the explanatory manuscript rewrite

**Decision: minor targeted content revision.** Read-only review of current10-reference version. No additional experiment is requested; this assesses claim support and reader comprehension, not acceptance probability.

The three contributions are now visible: empiricalcomponent diagnosis, a12-score case-disjoint benchmark, and a fixed-prediction endpoint-sensitivity test. The draft correctly presents these as analysis/evaluation rather than newVI mathematics. Old internalrun labels and irrelevantfusion narratives have been removed.

## R1 — P1

**Location:** Abstract final sentences; Methods Statistics and Review Policies; Discussion paragraph2

The draft retains an empirical fixed-budget-review conclusion but Results no longer exposes a corresponding numerical estimate/interval. A reader cannot assess how uncertain the asserted lack of reliable benefit is.

**Action:** Add one concise saved CT-budget estimate and CI (e.g. guard+partition,20%CT,missing-volume delta -0.00429,CI[-0.01213,0.00810]) with direction explained, or remove this empirical claim from the abstract/main framing. Do not add new experiments.

## R2 — P1

**Location:** Methods Mechanism Tests and Failure Endpoints; Results controlled responses

The first contribution depends on controlled interventions, yet the current Methods omits their severity levels and the result says boundary targets met a declared tolerance without stating it. This leaves the mechanism experiment harder to reproduce than necessary.

**Action:** Use one compact sentence specifying1–3 disjoint nearest-fragment merges,largest-fragment2–4-way splits and boundaryDice targets.99/.95/.90 with tolerance.005, as actually implemented; otherwise remove the tolerance result. Keep that this is controlledGT-copy algebra rather than realistic fracture generation.

## R3 — P2

**Location:** Methods Data, Models, and Evaluation Units

Calling a fourth segmentation network the additional reference model risks confusing it with the reference annotation/GT, especially because both occur throughout the Methods. Its role is to supply a different model-error target, not an independent reference standard.

**Action:** Name it a fourth segmentation model or separate segmentation model and say scores stay fixed while risks are recomputed from that model. Keep same-backend limitation.

## R4 — P2

**Location:** Introduction contribution list; Discussion/Conclusion

The component formulas and normalization properties are established quantities, so the contribution is empiricaldiagnosis and endpoint-controlled evidence. The draft mostly communicates this correctly; an occasional practical recommendation could still sound universal beyond the onebackend/casecollection.

**Action:** Retain the explicit no-new-mathematical-distance statement and use in-this-setting wording for the strongest recommendations. Explain that aCI crossing0 under a newendpoint shows no clear advantage, not equivalence or validated superiority.

## Verification scope

The current numericaltext/tables have saved-source support; no proposal or unfinishedE6 outcome is asserted as executed. Sourceauthor-name corrections from the earlier integrity pass remain fixed. The real-dataoverview was not yet final at this review snapshot, so its data/caption fidelity needs focused verification once available. These comments concern presentation of existing evidence, not a demand for stronger training results or more datasets.
