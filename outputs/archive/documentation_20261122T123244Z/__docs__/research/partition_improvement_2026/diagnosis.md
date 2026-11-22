# Partition: diagnosis before proposing a new method

**Date:** 22 November 2026. **Status:** read-only diagnosis and proposals, not a new experiment result or adopted protocol. No model was fit, no candidate was evaluated on held-out data, no bootstrap was rerun, and no existing code/config/output was changed. The new summaries below use only the 80 validation bone units and tiny synthetic arrays.

## 1. The current weakness is not an absent partition signal

The saved E1/E3 comparison already shows that partition-v1 is the strongest listed pooled merge/split score: validation AURC 0.2706 and held-out 0.2720. FA-v2 is 0.3318/0.3311. Those held-out values are existing findings, not a new search over candidates. The question should be why the combination underuses a strong signal, and whether a different structural representation can improve beyond that strong component control.

Sources: [current synthesis](../../reports/e1_e5_synthesis.md), `outputs/stage3_h1_v2/validation/comparison.csv`, and the recorded E3 main table.

## 2. Exact algebra of the implemented score

For one member pair, let `U` be the foreground union, `C` its intersection, `A=|U−C|/|U|`, and `w=|C|/|U|`. For a nonempty union, `w=1−A`. Let `Q_H=VI/H(P,Q)` on `C`, and `P_H=w Q_H`. The implemented score is

`FA_v2 = A + P_H = A + (1−A) Q_H`.

Thus increasing presence disagreement can increase the total score even when it adds no information about merge/split risk. For a nonempty union its sensitivity to `Q_H` is `1−A`; foreground mismatch simultaneously raises the score and reduces the coefficient of pure partition disagreement. Both-empty foreground has its separate zero convention.

The final bone score averages three pair scores: `mean(FA_v2)=mean(A)+mean(w Q_H)`. In general this is **not** `mean(A)+(1−mean(A))*mean(Q_H)`, because the covariance between pair components matters. Any proposed recalibration must preserve pair-versus-unit aggregation explicitly.

Implementation anchors: [e1_scoring.py](../../../src/exchange_ml/e1_scoring.py), lines 24–36 (information), 59–63 and 82–90 (weights/score), 107–127 (three-pair means). The validation CSV identity `fa_v2−presence−partition_v2` has maximum absolute residual `1.11e−16`.

## 3. Validation-only evidence: the channels predict different endpoints

The following are descriptive Spearman correlations over the existing 80 validation units, not inference from independent 80 patients. No confidence intervals or new significance tests are claimed.

| Score/component | Correlation with member-mean M+S | Correlation with member-mean missing volume |
|---|---:|---:|
| Presence A | 0.0317 | 0.5086 |
| Weighted partition-v1 | 0.7590 | 0.1015 |
| Weighted partition-v2 | 0.6920 | 0.0188 |
| FA-v2 | 0.5691 | 0.2056 |
| Raw mean-pair VI | 0.7593 | Not newly summarized here |

For M+S, pelvic/femur correlations are 0.0651/−0.2620 for presence, 0.8514/0.2878 for partition-v1, and 0.7869/0.2798 for partition-v2. There are only 20 femur units; the contrast is descriptive and does not justify tuning a femur-specific rule. Pooled correlations can also reflect anatomy and risk-distribution differences. Saved controlled-rank results remain more cautious: pooled partial rho for P1/P2/FA-v2 is 0.5773/0.5335/0.3951 after the existing GT-count/volume covariates; these covariates are evaluation controls, not legitimate deployment score inputs.

A further check within each anatomy prevents overgeneralizing the pooled near-zero presence association. With 20 units per bone, presence rho is 0.1588 (sacrum), 0.6238 (left hip), 0.3967 (right hip), and −0.2620 (femur). FA-v2 rho is 0.7186/0.8146/0.4292/0.0305, versus P2 0.9128/0.8632/0.4262/0.2798. Thus presence is not uniformly useless, and removing it everywhere is not a proven universal solution. Anatomy differences are a hypothesis for controlled development work, not permission to fit four unconstrained models on 20 units each.

Adding A changes 206 orderings among 2,688 unordered validation bone pairs with unequal risks and non-tied P2 and FA-v2. Of those, 166 change from correct ordering under P2 to incorrect under FA-v2; 40 change the opposite way. These dependent pair counts are an explanation of ranking disruption, **not 2,688 independent trials**, an AURC replacement, or proof that a new fitted combiner will generalize.

Two examples, selected to illustrate a recorded reversal rather than to estimate prevalence:

| Validation bone | A | P2 | FA-v2 | M+S risk |
|---|---:|---:|---:|---:|
| 030/sacrum | 0.15175 | 0.58167 | 0.73342 | 0.33333 |
| 193/sacrum | 0.02226 | 0.59583 | 0.61809 | 2.66667 |

P2 places the higher-risk second bone above the first, but adding A reverses them. Another recorded pattern is 047/sacrum: A=0.09482, P2=0, risk=0; it outranks 268/femur after addition although the latter has P2=0.03495 and risk=1.75.

Among 27 validation units with raw VI≤1e−12, only one has positive M+S risk (0.142857). This prevents an overstatement that the zero-partition mass generally hides structural failure in the validation cohort. It does not establish omission safety: complete misses are a different endpoint, already illustrated in E5.

## 4. Two distinct limits inside the partition representation

### 4.1 Joint-entropy normalization saturates for a one-cluster reference

If P contains one foreground cluster and Q splits it into k nontrivial clusters, then `H(P)=0`, `H(P,Q)=H(Q)` and `VI=H(Q)`. Consequently `VI/H(P,Q)=1` for every k>1. This is an algebraic property, not a coding error. It explains the saved E2 observation of 11 single-fragment bones with Q2=1 for all tested split severities.

Tiny example using the current function: 12 shared voxels, equal splits into k=2,3,4. Raw VI is 0.6931/1.0986/1.3863 nats, Q_H remains 1/1/1, and Q_logN is 0.2789/0.4421/0.5579. Normalization has removed this severity information. Merely removing overlap weight does not fix it; `q_joint_unweighted` is already a reported baseline and has essentially the same validation M+S AURC as P2.

### 4.2 Voxel-weighted information is not an object-count risk

The score's contingency counts weight voxels. The primary structural risk counts merge/split events and divides by GT instance count: [e1_data.py](../../../src/exchange_ml/e1_data.py), lines 81–97. In the internal evaluator a split is each additional positive predicted ID overlapping a GT instance: [instances.py](../../../src/exchange_ml/instances.py), lines 95–107. These are different measures of severity.

Exact tiny example with foreground fragments of 990 and 10 voxels, and no presence change:

| Perturbation | Count-normalized M+S | VI | Q_H |
|---|---:|---:|---:|
| Split the 10-voxel object into 5+5 | 0.5 | 0.006931 | 0.110140 |
| Split the 990-voxel object into 495+495 | 0.5 | 0.686216 | 0.924548 |

Both create one split among two original instances, but raw VI differs by a factor of 99. The example is intentionally unpruned synthetic data and does not estimate clinical error importance. It proves a representation/endpoint mismatch is possible, not that all observed ranking loss is caused by object size. Validation size-stratified diagnostics would be needed to attribute the empirical gap.

The current E2 split intervention always splits the largest GT fragment ([e2_perturbations.py](../../../src/exchange_ml/e2_perturbations.py), lines 136–154). Therefore E2 as implemented does **not** test sensitivity to equally counted small-versus-large-fragment errors. A predeclared size-swap phantom would be a useful diagnostic extension.

## 5. Three intervention families worth comparing

These are proposed research directions, not implemented options or favorable results. Any method derived after observing E3 must be labeled accordingly and needs a fresh confirmation set for a new confirmatory claim. Parent orchestration should check remaining-case provenance before naming that set.

### Priority 1: endpoint-specific channels and a small calibrated combiner

**Hypothesis:** fixed addition mixes signals that target different errors. Preserve a structural channel and a presence/omission channel rather than forcing their raw magnitudes into one scalar.

**Intervention:** start with the frozen component scores as separate inputs; compare a small regularized endpoint-specific calibration/combination with fixed FA-v2 and the strongest frozen partition control. Fit only on validation/development cases, using case-grouped out-of-fold evaluation and very few degrees of freedom. Presence is an optional input to the structural head, not a required coefficient of one. An omission head must target omission-sensitive risk. A two-queue review policy must still obey one total review/cost budget and handle duplicate bones/CTs; otherwise it unfairly spends more review capacity.

**Controls:** FA-v1/v2, P1/P2, unweighted Q_H, raw VI, presence, Hungarian, and a capacity-matched simple linear/rank combiner. A P1-only structural head is essential: merely reproducing P1's benefit does not establish a stronger joint detector.

**Readout:** paired out-of-fold AURC and fixed-budget retained M+S plus missing-volume risk, with case-level uncertainty and CT cost beside bone count. Success means improving on the relevant strong control without unacceptable omission/cost tradeoff; a win over FA-v2 alone is not enough. Compute is initially CPU on existing CSVs, but no fitting was run in this diagnosis.

### Priority 2: object-sensitive structural features rather than another denominator

**Hypothesis:** shared-voxel VI/H obscures counted structural events through size weighting and saturation.

**Intervention:** compare an object-overlap graph or object-macro conditional-entropy features computed from predictions alone. Pairwise one-to-many/many-to-one overlap summaries can distinguish structural multiplicity from foreground extent. Keep raw VI and the two conditional entropies as separate features; avoid immediately dividing away severity. Directional entropy describes splitting relative to the other prediction, **not which model is wrong against GT**. Shared-foreground restriction and empty rules must be explicit.

**Controls:** existing P1/P2, raw VI, count variance and Hungarian; the latter two already capture some object information. Predeclare a size-swap phantom, varying split multiplicity, tiny-overlap/boundary jitter, label permutations and count replication. Overlap thresholds must be fixed on development data because single-voxel contact can otherwise create many graph edges. Object balancing should be evaluated against count risk and volume risk separately; size invariance is not automatically appropriate for a volume-sensitive endpoint.

**Readout:** sensitivity to equal-count errors on differently sized fragments, avoidance of Q_H's one-cluster saturation, and development/fresh-confirmation ranking beyond P1. CSVs contain scalar VI/H but not all per-object contingency structure, so graph/macro features may need reading saved instance maps; this is a new CPU analysis, not a zero-cost CSV reweighting. No model retraining is intrinsically required.

### Priority 3: an independent omission safeguard

**Hypothesis:** disagreement alone cannot detect a common-mode omission. Identical correct and identical wrong instance maps can have the same zero disagreement; a deterministic function of those disagreement values cannot distinguish them.

**Intervention:** add a separate GT-free expected-anatomy check using scan-type/coverage metadata and predicted anatomy presence, or a genuinely independent image/anatomy signal. This can flag the known all-members-empty situation without pretending the partition score detects it. Expectations must be appropriate to scan coverage; anatomical variation or absent anatomy can create false alarms.

**Controls/readout:** unchanged structural ranking, the safeguard alone and the combined policy at equal total cost. Measure omission sensitivity, false alerts and additional reviewed CTs, while reporting M+S separately. This is safety coverage for another failure mode; it is not evidence of improved partition discrimination. It is also the supplement's previously proposed QC extension, not a newly validated result.

## 6. Reproducibility and limits

The diagnosis reads `validation/scores.csv`, `pair_components.csv`, `risks.csv` and `comparison.csv`; their SHA-256 values matched the existing E1 manifest:

| File | SHA-256 |
|---|---|
| scores.csv | `8e5717fca87842a067dff9d590ee74d9250d5ba04d3ce408266e53aaead25c01` |
| pair_components.csv | `7f80ab8ac0ff7e63603c66ea411984ceb991e6af9d2fc54a6bf23c177238c307` |
| risks.csv | `ccbe15e8de196edb32f1e34755f9db81c79e485818363b7f7e1eb5f746176df5` |
| comparison.csv | `f0dd12873aefe4e992c9d4f18734f8c61a4804e5137fe05b9cdcd85e59ef71b8` |

**Agent-typed checks:** Python CSV reads and NumPy/SciPy descriptive correlations/order counts; existing `pair_components` and `instance_metrics` on 12-voxel/1,000-voxel toy arrays; manifest hash comparison; source-line inspection. All CPU-only. **User-typed full-run commands:** none prepared. No bootstrap, learned weights, new held-out scoring or full-volume processing was executed.

The new validation summaries illustrate mechanisms; they are development diagnostics rather than independent confirmation. Current broad claims remain limited by endpoint selection, anatomy weighting and one backend family. New proposals must not overwrite the frozen E1–E6 protocols or turn post-held-out selection into a preregistered result.

### Reproduce the bounded diagnosis

The read-only helper [reproduce_diagnosis.py](reproduce_diagnosis.py) verifies the four E1 source-table hashes, reproduces the pooled/anatomy correlations and dependent-pair counts, and checks the small existing VI examples. It was run successfully with:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B \
  __docs__/research/partition_improvement_2026/reproduce_diagnosis.py
```

This reproduces descriptive validation evidence; it does not fit a new score, run a bootstrap or evaluate any candidate on held-out cases. The 990/10-voxel example deliberately disables component pruning and proves an algebraic mismatch only; it is not a simulation of the complete production cleanup pipeline. Multiplying all contingency counts preserves the VI and VI/H statements, but actual image resampling or cleanup can change the partitions.

The root agent also independently checked the proposed graph formula on tiny contingency matrices: balanced splits into 2/3/4 counterparts give 0.5/1/1.5, with identity, symmetry, count-replication and matched-relative-split size-swap checks passing. These are toy properties of a proposal, not real-data ranking gains.
