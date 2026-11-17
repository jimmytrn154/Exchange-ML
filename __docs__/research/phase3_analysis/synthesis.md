# Synthesis Report

Phase 3 · `synthesis_agent` · 2026-11-17 · inputs: `phase1_scoping/*`, `phase2_investigation/annotated_bibliography.md` (28 sources, slugs below), `phase2_investigation/source_verification.md`.

Slug table (corpus context): meila2007 · meila2003 · vinh2010 · arganda2015 · nunez2013 · kirillov2019 · lakshminarayanan2017 · jungo2019 · roy2019 · mehrtash2020 · zenk2025 · kahl2024 · geissler2024 · gupta2023 · yang2025 · berger2025 · jaeger2023 · geifman2019 · maierhein2024 · reinke2024 · phitidis2026 · sang2026 · liu2023 · raniga2025 · isensee2021 · huang2023 · shi2021 · abbc2026.

## Claim Intent Manifest (R-CIM-A; emitted before the first prose block; no passport file in this run, so recorded inline)

```json
{
  "manifest_version": "1.0",
  "manifest_id": "M-2026-11-17T00:00:00Z-s3a1",
  "emitted_by": "synthesis_agent",
  "emitted_at": "2026-11-17T00:00:00Z",
  "claims": [
    {"claim_id": "C-001", "claim_text": "VI normalised by log N or 2 log K has a narrower useful range than VI normalised by the joint entropy H(U,V), which is itself bounded by 2 log K when N exceeds the number of contingency cells.", "intended_evidence_kind": "theoretical", "planned_refs": ["vinh2010", "meila2007"]},
    {"claim_id": "C-002", "claim_text": "Information-theoretic segmentation scores have been split into under- and over-segmentation (split/merge) components and restricted to foreground in the connectomics literature.", "intended_evidence_kind": "empirical", "planned_refs": ["arganda2015", "nunez2013"]},
    {"claim_id": "C-003", "claim_text": "Pairwise overlap disagreement between ensemble members is a strong, simple case-level failure-detection baseline for semantic segmentation.", "intended_evidence_kind": "empirical", "planned_refs": ["zenk2025", "geissler2024", "roy2019", "lakshminarayanan2017"]},
    {"claim_id": "C-004", "claim_text": "The term 'structural uncertainty' denotes different constructs across recent work (topological, shape-model, instance-partition).", "intended_evidence_kind": "empirical", "planned_refs": ["gupta2023", "yang2025", "berger2025"]},
    {"claim_id": "C-005", "claim_text": "Risk-coverage analysis / AURC is the recommended primary evaluation for failure detection, and aggregation and confounders (structure size, prevalence, partial annotation, matching rules) are documented pitfalls.", "intended_evidence_kind": "empirical", "planned_refs": ["jaeger2023", "geifman2019", "zenk2025", "kahl2024", "geissler2024", "maierhein2024", "reinke2024", "phitidis2026", "sang2026"]},
    {"claim_id": "C-006", "claim_text": "Pelvic fracture segmentation underpins fragment-wise reduction planning; fragment delineation on CT, not fragment count, is what the clinical literature ties to surgical decisions.", "intended_evidence_kind": "empirical", "planned_refs": ["sang2026", "raniga2025"]},
    {"claim_id": "C-007", "claim_text": "No retrieved study measures ensemble disagreement over instance partitions against merge/split error in 3D medical CT (bounded absence claim).", "intended_evidence_kind": "empirical", "planned_refs": ["zenk2025", "kahl2024", "gupta2023"], "negative_constraints": [{"constraint_id": "NC-C007-1", "rule": "State the absence only within the documented search scope; never claim 'first' or 'no prior work'."}]},
    {"claim_id": "C-008", "claim_text": "On the project's 40 validation cases the v1 partition term has median share 2 percent of FA-IPD and FA-IPD v1 is rank-identical to foreground Jaccard (rho 0.997).", "intended_evidence_kind": "empirical", "planned_refs": [], "planned_experiment_ids": ["h1-v1-2026-10-21"]}
  ],
  "manifest_negative_constraints": [
    {"constraint_id": "MNC-1", "rule": "No active-learning, acquisition or RibFrac claims."},
    {"constraint_id": "MNC-2", "rule": "No claim that FA-IPD v2 outperforms anything; v2 is untested until the held-out run."},
    {"constraint_id": "MNC-3", "rule": "No 'new metric' claim; FA-IPD is a dissimilarity score."}
  ]
}
```

## Literature Matrix

| Source | A: partition measures | B: ensemble / structural uncertainty | C: evaluation method | D: task / backend | Method | Level |
|---|---|---|---|---|---|---|
| meila2007 | Core (VI, metric, bounds) | — | — | — | Theory | VII |
| meila2003 | Core | — | — | — | Theory | VII |
| vinh2010 | **Core** (NVI, NID, log N vs H(U,V)) | — | — | — | Theory | VII |
| arganda2015 | Core (split/merge, foreground-restricted) | — | Supports (metric sensitivity) | — | Challenge | VI |
| nunez2013 | Supports (VI at voxel scale) | — | — | — | Method | VI |
| kirillov2019 | Supports (IoU matching) | — | Supports | — | Method | VI |
| lakshminarayanan2017 | — | Core (ensembles) | — | — | Method | VI |
| jungo2019 | — | Supports (subject-level miscalibration) | Supports | — | Benchmark | VI |
| roy2019 | — | Core (structure-wise, MC-IoU ≈ Dice) | — | — | Method | VI |
| mehrtash2020 | — | Supports | — | — | Method | VI |
| zenk2025 | — | **Core** (pairwise Dice best) | **Core** (risk-coverage) | — | Benchmark | VI |
| kahl2024 | — | Supports (aggregation C3) | Core | — | Framework | VI |
| geissler2024 | — | Supports (mean pairwise Dice) | **Core** (size confounder) | — | Empirical | VI |
| gupta2023 | — | Diverges (topological construct) | — | — | Method | VI |
| yang2025 | — | Diverges (shape construct) | — | — | Method | VI |
| berger2025 | — | Supports | Core (pitfalls) | — | Empirical | VI |
| jaeger2023 | — | — | **Core** (AURC) | — | Benchmark | VI |
| geifman2019 | — | — | Core (AURC origin) | — | Method | VI |
| maierhein2024 | — | — | Core | — | Consensus | VII |
| reinke2024 | — | — | Core | — | Consensus | VII |
| phitidis2026 | — | — | Supports (partial-label scope) | — | Empirical | VI |
| sang2026 | Supports (highest-IoU matching) | — | — | **Core** | Challenge | VI |
| liu2023 | — | — | — | Supports | Method | VI |
| raniga2025 | — | — | — | Core (clinical) | Review | VII |
| isensee2021 | — | — | — | Supports | Method | VI |
| huang2023 | — | — | — | Supports | Preprint | VI |
| shi2021 | — | — | — | Supports (partial-label loss) | Method | VI |
| abbc2026 | — | — | — | Supports | Software | grey |

## Key Themes

### Theme 1 — The normalisation of VI decides whether partition disagreement is measurable at voxel scale
**Evidence strength:** Strong (theory, primary text read). **Sources:** 4, Levels VI–VII.

Variation of information is a metric on partitions with VI ≤ log N for N elements and VI ≤ 2 log K when both partitions have at most K parts — Meilă (2007) <!--ref:meila2007--><!--anchor:section:abstract-->. Vinh et al. (2010) <!--ref:vinh2010--><!--anchor:section:3.3--> compare the normalisations explicitly: the joint-entropy form `NVI = VI / H(U,V)` is a metric (their Theorem 3), whereas Meilă's `VI/log N` and `VI/2 log K` rest on bounds that "are not as strict as H(U,V)", so "the useful range of these normalized VI variants is narrower"; their footnote 4 gives the chain `H(U,V) ≤ log(RC) ≤ 2 log K` whenever N exceeds the number of contingency cells. Read against the project's setting — N ≈ 10⁵–10⁶ shared-foreground voxels, K ≤ 12 fragments — this is not a subtlety: `log N ≈ 12–14` while `H(P,Q) ≤ 2 log 12 ≈ 5`, so the v1 term `VI/log N` can never exceed ≈ 0.36 and in practice sits far lower because fragment partitions mostly agree (observed median 0.0034, max 0.0198 on the 40 validation cases; experiment `h1-v1-2026-10-21`). The two presence-type terms in FA-IPD (symmetric difference / union) are bounded by 1 and routinely reach 0.4, so the sum is a presence score. Vinh et al. also raise a caveat the paper must carry: data-dependent normalisers such as H(U,V) have "potential side effects on the normalization process" because the normaliser depends on both partitions <!--ref:vinh2010--><!--anchor:section:3.3-->; with both partitions being predictions the asymmetry they discuss (ground truth vs prediction) does not arise, but H(P,Q) still co-varies with fragment count, which routes the issue into Theme 4's confounder control. They further note NID "is generally more preferable" to NVI for using the [0,1] range <!--ref:vinh2010--><!--anchor:section:3.3-->; the project's choice of NVI (continuity with the VI-based v1 term; VI decomposes into H(P|Q)+H(Q|P), the split/merge pair) is defensible but must be stated, with NID named as the untested alternative.

The connectomics challenge literature supplies the structural precedents. Arganda-Carreras et al. (2015) <!--ref:arganda2015--><!--anchor:section:2.5--> score segmentations with information-theoretic **split** `I(S;T)/H(S)` and **merge** `I(S;T)/H(T)` components combined as an F-score "closely related to the variation of information", and restrict Rand scoring to foreground because border widths are arbitrary <!--ref:arganda2015--><!--anchor:section:2.4-->; Nunez-Iglesias et al. (2013) <!--ref:nunez2013--><!--anchor:section:abstract--> advocate VI for 3D EM instance segmentation. The FA-IPD construction — presence term plus a foreground-restricted partition term — is therefore a recombination of established pieces; what the corpus does not contain is any analysis of how those pieces scale when the partition term is normalised by `log N` on voxel-count-sized domains.

### Theme 2 — Ensemble overlap disagreement is a strong failure-detection baseline, and it is a presence signal
**Evidence strength:** Strong–moderate (benchmark + method papers, abstract level). **Sources:** 6, Level VI.

Deep ensembles (Lakshminarayanan et al., 2017) <!--ref:lakshminarayanan2017--><!--anchor:section:abstract--> are the standard disagreement generator; Mehrtash et al. (2020) <!--ref:mehrtash2020--><!--anchor:section:abstract--> report improved calibration from them in medical segmentation. Three converging results show that **overlap between members** is the useful case-level statistic: Roy et al. (2019) <!--ref:roy2019--><!--anchor:section:abstract--> find IoU across MC samples a suitable Dice proxy per structure; Geißler et al. (2024) <!--ref:geissler2024--><!--anchor:section:abstract--> use mean pairwise Dice as a structure-wise measure; and the five-dataset benchmark of Zenk et al. (2025) <!--ref:zenk2025--><!--anchor:quote:superior%20performance%20of%20the%20pairwise%20Dice%20score--> finds pairwise Dice "a simple and robust baseline for failure detection", with "pixel confidence aggregation" as the decisive design choice. Jungo and Reyes (2019) <!--ref:jungo2019--><!--anchor:section:abstract--> explain why voxel-wise measures need such aggregation: they are miscalibrated at subject level.

Every one of these signals is a **presence/overlap** disagreement on a semantic target. The project's v1 result (FA-IPD ≈ foreground Jaccard, ρ = 0.997; experiment `h1-v1-2026-10-21`) is therefore not a surprise relative to this theme — it reproduces Zenk et al.'s conclusion inside an instance task and shows that an additive structural term cannot be seen next to it under v1 scaling. What the theme leaves open is whether, once scaled, a partition term carries information about **instance** failures that overlap disagreement does not.

### Theme 3 — "Structural uncertainty" names three different constructs
**Evidence strength:** Emerging (divergent definitions). **Sources:** 3, Level VI.

Gupta et al. (2023) <!--ref:gupta2023--><!--anchor:section:abstract--> estimate uncertainty over topological structures of curvilinear objects via discrete Morse theory (inter- vs intra-structural); Yang et al. (2025) <!--ref:yang2025--><!--anchor:section:abstract--> derive "structural uncertainty" from an active-shape-model fit to a single object; Berger et al. (2025) <!--ref:berger2025--><!--anchor:section:abstract--> treat topology-aware segmentation and show that connectivity choice, ground-truth artifacts and metric choice change method rankings. None of the three addresses a variable-cardinality partition of one bone into fragments. The project's term must therefore be defined on first use as *instance-partition* disagreement and distinguished from topological and shape-model senses; "topology-aware" should not be used (consistent with `__docs__/paper.md` §2.1).

### Theme 4 — Evaluation design for failure detection: AURC plus four documented confounders
**Evidence strength:** Strong (consensus + benchmark). **Sources:** 9, Levels VI–VII.

Jaeger et al. (2023) <!--ref:jaeger2023--><!--anchor:section:abstract--> argue for AURC as the primary failure-detection metric, separate from calibration, building on the selective-prediction formulation of Geifman et al. (2019) <!--ref:geifman2019--><!--anchor:section:abstract-->; Zenk et al. (2025) <!--ref:zenk2025--><!--anchor:section:abstract--> carry risk-coverage analysis into medical segmentation; Kahl et al. (2024) <!--ref:kahl2024--><!--anchor:section:abstract--> make aggregation an explicit component (C3) of any uncertainty method. Each of the project's four evaluation conditions has a semantic-level precedent: (a) score/risk from different models — implicit in ValUES' insistence that C1–C3 be ablated together <!--ref:kahl2024--><!--anchor:section:abstract-->; (b) prevalence-dependent metrics are a named pitfall class in Reinke et al. (2024) <!--ref:reinke2024--><!--anchor:section:abstract--> and Maier-Hein et al. (2024) <!--ref:maierhein2024--><!--anchor:section:abstract-->; (c) partially labelled evaluation data "inherently limits our capacity to systematically quantify false-positive predictions for unannotated classes" — Phitidis et al. (2026) <!--ref:phitidis2026--><!--anchor:section:5%20Limitations-->; (d) matching rules: PENGWIN 2024 matched "each ground-truth fragment … to the predicted fragment with the highest IoU" with no minimum threshold <!--ref:sang2026--><!--anchor:section:Evaluation-->, and Kirillov et al. (2019) <!--ref:kirillov2019--><!--anchor:section:abstract--> fix IoU ≥ 0.5 for panoptic matching — the project's IoU ≥ 0.1 sits between these and is the reason condition (d) exists. Structure size as a confounder of uncertainty–quality correlation, with partial correlation as the remedy, is Geißler et al. (2024) <!--ref:geissler2024--><!--anchor:section:abstract-->. The synthesis point: the ingredients of a sound instance-level failure-detection evaluation exist, but they are scattered across classification, semantic segmentation and challenge-design papers; no retrieved work assembles them for instance-level structural failures (bounded absence, see Gap 2). One lineage caveat: AURC, the pairwise-Dice baseline, ValUES and Metrics Reloaded share a DKFZ/Heidelberg authorship core (verification report, COI section), so the protocol follows one school rather than independent consensus.

### Theme 5 — Task and clinical framing
**Evidence strength:** Moderate. **Sources:** 6, Levels VI–VII plus grey.

The PENGWIN 2024 report states that "automated pelvic fracture segmentation is a crucial step in trauma diagnosis and surgical planning" and that "precise fracture segmentation is a prerequisite for fragment-wise reduction planning" <!--ref:sang2026--><!--anchor:quote:precise%20fracture%20segmentation%20is%20a%20prerequisite%20for%20fragment-wise%20reduction%20planning-->; its CT set had 150 scans with 3–12 fragments per case and sacrum / left hip / right hip labels <!--ref:sang2026--><!--anchor:section:Dataset-->. The imaging review of Raniga et al. (2025) <!--ref:raniga2025--><!--anchor:section:abstract--> ties CT delineation of fragment rotation, comminution and displacement to classification and reduction planning; no retrieved source makes fragment **count** a surgical cutoff, so the paper's clinical sentence must speak of fragment delineation and displacement. The backend lineage is FracSegNet (Liu et al., 2023) <!--ref:liu2023--><!--anchor:section:abstract-->, nnU-Net (Isensee et al., 2021) <!--ref:isensee2021--><!--anchor:section:abstract-->, STU-Net (Huang et al., 2023, preprint) <!--ref:huang2023--><!--anchor:section:abstract-->, the marginal loss for partial labels (Shi et al., 2021) <!--ref:shi2021--><!--anchor:section:abstract--> and the ABBC pipeline <!--ref:abbc2026--><!--anchor:none:-->. The project's dataset is the 2026 edition (340 cases) and has no peer-reviewed description in the corpus.

## Contradictions & Resolutions

| Claim A | Claim B | Resolution |
|---|---|---|
| zenk2025: pairwise Dice is the superior failure-detection baseline | Project hypothesis: a scaled partition term adds signal | **Conditional difference.** Zenk's targets are semantic; the project's risks are instance-level merge/split. Reconciled by scoping C2 to merge+split risk and reporting pairwise Dice/Jaccard as the baseline to beat (¶ Theme 2). |
| vinh2010: NID "generally more preferable" to NVI | Blueprint V1 fixes NVI = VI/H(P,Q) | **Conditional difference.** NVI retained for continuity with the VI-based v1 term and its split/merge decomposition; NID recorded as the untested alternative. Flagged for the author's pre-registration decision (¶ Theme 1). |
| vinh2010: data-dependent normalisers have side effects | Blueprint uses H(P,Q) (data-dependent) | **Conditional difference, resolved in design.** Both partitions are predictions (symmetric case); residual dependence on fragment count handled by partial correlation (¶ Theme 1, Theme 4). |
| roy2019: structure-wise uncertainty highly correlated with Dice | geissler2024: such correlations are inflated by structure size | **Conditional difference, resolved.** Geißler's partial correlation is the remedy; the project adopts it (¶ Theme 4). |
| arganda2015 / nunez2013: VI-type scores work for EM instance segmentation | Project v1: VI/log N term is negligible | **No material conflict.** Those works use VI unnormalised or I/H-normalised, not VI/log N; the v1 collapse is a normalisation effect, not a VI failure (¶ Theme 1). |

#### Cross-Paper Tension Inventory (#262)

```yaml
cross_paper_tensions:
  - pair_id: CP-001
    paper_a: "zenk2025"
    paper_b: "geissler2024"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "mean/pairwise Dice disagreement as a structure- or case-level quality predictor"
    a_finding: "pairwise Dice is a superior, robust failure-detection baseline"
    a_evidence_pointer: "annotated_bibliography #11, abstract quote"
    b_finding: "size confounds uncertainty–quality correlations incl. mean pairwise Dice; partial correlation gives lower, truer estimates"
    b_evidence_pointer: "annotated_bibliography #13, PMLR abstract"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 4; Theme 4"
    scholar_confirmation: "pending"
  - pair_id: CP-002
    paper_a: "vinh2010"
    paper_b: "meila2007"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "normalisation of VI"
    a_finding: "VI/log N and VI/2 log K have narrower useful range than VI/H(U,V); NVI and NID are metrics"
    a_evidence_pointer: "annotated_bibliography #3, §3.3 and footnote 4 (primary text)"
    b_finding: "proposes VI and its log N / 2 log K bounds"
    b_evidence_pointer: "annotated_bibliography #1, abstract; as cited in vinh2010 §3.3"
    pair_assessment: "no_material_conflict"
    resolution_status: "not_applicable"
    scholar_confirmation: "pending"
  - pair_id: CP-003
    paper_a: "gupta2023"
    paper_b: "yang2025"
    candidate_basis: "shared RQ subtopic"
    overlap_topic: "definition of structural uncertainty"
    a_finding: "uncertainty over topological structures via discrete Morse theory"
    a_evidence_pointer: "annotated_bibliography #14, abstract"
    b_finding: "uncertainty from active-shape-model fitting"
    b_evidence_pointer: "annotated_bibliography #15, abstract"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Theme 3"
    scholar_confirmation: "pending"
  - pair_id: CP-004
    paper_a: "sang2026"
    paper_b: "kirillov2019"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "instance matching rule"
    a_finding: "GT fragment matched to the predicted fragment with the highest IoU (no threshold stated)"
    a_evidence_pointer: "annotated_bibliography #22, Evaluation section"
    b_finding: "panoptic matching requires IoU > 0.5"
    b_evidence_pointer: "annotated_bibliography #6, abstract-level knowledge of PQ definition"
    pair_assessment: "conditional_difference"
    resolution_status: "flagged_unresolved"
    scholar_confirmation: "pending"
  - pair_id: CP-005
    paper_a: "jaeger2023"
    paper_b: "reinke2024"
    candidate_basis: "bibliographic coupling"
    overlap_topic: "choice of failure-detection / validation metric"
    a_finding: "AURC as primary failure-detection metric"
    a_evidence_pointer: "annotated_bibliography #17, abstract"
    b_finding: "metric choice has prevalence- and threshold-related pitfalls"
    b_evidence_pointer: "annotated_bibliography #20, abstract"
    pair_assessment: "no_material_conflict"
    resolution_status: "not_applicable"
    scholar_confirmation: "pending"
```

**Coverage Note:** 28 papers in corpus; 5 candidate pairs considered (basis: shared construct, shared RQ subtopic, bibliographic coupling). This is a scoped advisory scan, not complete pairwise contradiction detection — cross-neighbourhood pairs not surfaced here may exist and are not claimed absent. Bibliographic coupling was used as an inclusion signal only. Scholar confirms each `resolution_pointer` and may flag additional cross-pairs.

## Knowledge Gaps

1. **Empirical (bounded absence).** Within the documented search (themes B and D, 2019–2026), no study measures ensemble disagreement over **instance partitions** against merge/split or missing-fragment error in 3D medical CT; the nearest work scores semantic overlap (zenk2025, geissler2024, roy2019). Implication: C2 is a genuine open question, not a replication. (NC-C007-1: say "not found in this scope", never "first".)
2. **Methodological.** No assembled evaluation protocol for instance-level structural failure detection that fixes the unit of analysis (per annotated structure), controls prevalence, matches score and risk models, and declares the matching rule's blindness to partial loss. Each piece exists separately (Theme 4). Implication: C3 fills a protocol gap, and the paper should present it as an assembly with sources, not an invention.
3. **Theoretical.** The scale behaviour of VI normalisations at N ≫ K is stated in vinh2010 for clusterings but has not been applied to voxel-partition disagreement in any retrieved segmentation paper. Implication: C1's bound is a small but real theoretical contribution; present it as an application of vinh2010, with the numeric regime (N ≈ 10⁵–10⁶, K ≤ 12) as the new content.
4. **Dataset.** No peer-reviewed description of the PENGWIN 2026 edition (340 cases) was found. Implication: dataset facts cite the project audit and the challenge site; the 2024 report is cited for lineage only.
5. **Clinical.** No retrieved source links fragment-count errors in automated segmentation to planning or outcome measures. Implication: the clinical motivation stays at "fragment delineation and displacement on CT guide planning" (raniga2025, sang2026); no outcome claim.

## Evidence Convergence Map

```
Strong:      [==========] Theme 1  VI normalisation & scale        (4 sources, VI–VII; primary text read)
Strong:      [=========-] Theme 4  AURC + confounders               (9 sources, VI–VII; abstract level)
Moderate+:   [=======---] Theme 2  overlap disagreement as baseline (6 sources, VI; abstract level)
Moderate:    [=====-----] Theme 5  task & clinical framing          (6 sources, VI–VII + grey)
Emerging:    [===-------] Theme 3  'structural uncertainty' senses  (3 sources, VI; divergent)
Gap:         [----------] instance-partition disagreement vs merge/split error in 3D CT (0 sources in scope)
```

## Theoretical Integration

Three frameworks hold the synthesis together. (i) In the ValUES decomposition (kahl2024), FA-IPD is not a new uncertainty *method* but a choice of **uncertainty measure + aggregation (C2 + C3)** operating on instance maps rather than voxels; this positions the paper correctly and keeps it clear of "new metric" language. (ii) Vinh et al.'s normalisation theory (vinh2010) supplies the mechanism for C1 and the pre-registered form for C2; the paper applies, rather than extends, that theory. (iii) The Metrics Reloaded problem fingerprint (maierhein2024) — instance segmentation, variable cardinality per image, small structures, partially annotated images — predicts exactly the pitfalls enumerated in C3 and gives the paper a recognised vocabulary for them. Together they make the paper's claim shape clear: an **analysis** of what an existing disagreement signal measures under stated scaling and evaluation conditions, confirmed (or not) on held-out data.

## Synthesis Limitations

- 22 of 28 sources were integrated at abstract level; claims from those sources are abstract-level and anchored as such.
- Contradiction detection is a scoped scan (5 pairs), not exhaustive.
- The empirical statements about the project's own data (C-008) come from `outputs/stage3_h1/` artifacts dated 2026-10-21 and are the scholar's own results, not literature.
- The bounded-absence claim (Gap 1) rests on web search plus OpenAlex/Crossref; a Scopus/WoS sweep could surface instance-level work not found here.
- One intellectual lineage (DKFZ/Heidelberg) dominates Theme 4; the synthesis notes but cannot remove this.
