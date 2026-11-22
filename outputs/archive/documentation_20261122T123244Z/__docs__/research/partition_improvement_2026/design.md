# Partition improvement: defensible experiment design

**Status: proposal only, 22 November 2026.** Scope: research-design assessment requested through `academic-pipeline`. No model was fitted, no candidate was scored on held-out outcomes, and no existing code, config, checkpoint or result was changed.

## 1. Define the problem before choosing a remedy

The existing partition signal is not globally weak: pooled held-out merge/split AURCs are 0.2720 for partition-v1 and 0.2978 for partition-v2, against 0.3311 for FA-v2. The useful question is whether one can improve **the combination, severity response and omission coverage**, while retaining the structural information already carried by the strongest component. A remedy must be compared with partition-v1, partition-v2, FA-v2 and the existing Hungarian/count controls, not just entropy or presence.

Two goals require separate success criteria:

1. A better **structural ranker** must improve the same normalized merge/split endpoint relative to the strongest partition-only control.
2. A better **review policy** may retain structural ranking while covering omission failures at a fixed review cost. Such a policy need not beat partition-only on merge/split AURC alone; it must demonstrate the claimed trade-off using omission-sensitive outcomes and equal cost.

These are future extensions. The existing E3 primary comparison and E1–E5 results stay unchanged. A better retrospective score on the already inspected 100 E3 cases would be exploratory evidence, not a new confirmation.

## 2. Data-role audit: a possible fresh confirmation set exists, conditionally

Read-only set arithmetic on `splits/pengwin_stage0_split.json` and `configs/stage3_h1_v2.json` gives:

| Role | Cases |
|---|---:|
| Static pool | 250 |
| Trajectory-20260920 initial labeled set | 40 |
| Existing E3 held-out cases | 100 |
| Pool minus initial train minus E3 | **110** |
| Separate diagnostic validation | 40 |
| Original reserved test set | 50 |

The remaining 110 comprise **55 pelvic and 55 femur cases**, verified using the existing directory-based scan-type inventory, not only numeric-ID guesses. They have zero overlap with the initial training, diagnostic validation, original test and existing E3 case sets. Recorded nnU-Net splits show Anatomy train/validation 40/40 subject samples and Fragment 80/80 anatomy-specific samples. `configs/stage3_h1.json` points all three seed members to the same Stage 2 data preparation and trajectory.

No remaining-case ID appeared in the active `scores.csv`, `risks.csv`, `model_risks.csv`, case-risk tables, `uncertainty_scores.csv` or `validation_metrics.csv` inspected. This is **candidate eligibility, not proof of untouched scientific outcomes**: archived copies, unrecorded prior inspection, collaborator analyses and notebook history were not exhaustively audited. The Stage 0 split itself explicitly used GT-derived static strata, so “GT has never been read” would also be inaccurate.

Before allocating the 110 as new confirmation data, audit their inference/evaluation history and obtain an explicit no-outcome-exposure record. Freeze their list, candidate implementation, fitting rule and decision criteria before processing them. If that audit fails, find genuinely new cases; do not use the existing 100 E3 cases as a substitute fresh test. Keep the 50 originally reserved test cases in their established role.

## 3. First candidate: learn a small structural-risk fusion instead of a fixed sum

**Hypothesis.** Presence and partition may contain useful but endpoint-dependent information; an uncalibrated positive sum can inject a component that is poorly aligned with merge/split severity. Let the development data determine whether presence adds useful information beyond the structural reference, without presupposing its coefficient must be positive.

**Target.** Mean of the three members' separately evaluated normalized merge/split risk, using the unchanged internal evaluator. GT is permitted for fitting and validation labels, never as an inference-time feature. It remains a risk-ranker score, not a newly calibrated probability or mathematical metric.

**Intervention.** Begin with only two prediction-derived features: partition-v1 and presence. A regularized linear model for `log1p(risk)` is sufficient as an initial hypothesis; its output is used for ordering, not interpreted as a clinical probability. Standardize using training-fold statistics only. An optional second development candidate can add predicted fragment count and foreground-volume summaries; use prediction-derived values, not `n_GT`, GT volume or smallest GT fragment volume. Do not start with a random forest or flexible neural combiner on 40 cases.

**Controls.** All existing frozen scores; a partition-only ranker/calibration; a metadata/predicted-size-only model if those features are added; and the same two-feature model without presence. A monotonic transformation of partition-v1 cannot improve its within-stratum ordering except ties: it is a calibration control, not a claimed ranking innovation. A learned model that merely recovers partition-v1 is an informative negative result rather than a new method benefit.

**Development.** The 40 diagnostic cases yield 80 bones but only 40 case groups. Use five outer case-grouped folds stratified by scan type, keeping every bone from a CT in one fold. If choosing regularization, feature subsets or policy thresholds, perform those choices in inner grouped folds. Limit the candidate family and regularization grid before evaluation. All scaling, feature selection and thresholding belong inside the relevant training fold. Aggregate out-of-fold predictions before evaluating the development procedure; do not select a candidate using the already exposed E3 outcomes.

**Metrics.** Existing tie-aware per-bone merge/split AURC; paired case-cluster uncertainty versus partition-v1 as the new extension's predeclared main contrast; partition-v2 and all other controls remain reported. Check pelvic/femur separately and both existing risk sources. Any minimum meaningful effect must be chosen before fresh outcomes; do not silently inherit the old E3 effect floor as a new registered criterion.

**Interpretation.** A gain over entropy but no gain over partition-v1 does not solve the weakness. A pooled-only gain with deteriorating strata or omission risk supports a narrower task-specific claim. A gain that disappears against the metadata-only control suggests case-type/size risk prediction rather than improved partition information.

**Compute.** Development uses existing validation CSVs and CPU; no base segmentation retraining is required. The full cross-validation/analysis remains a user-run experiment under repository rules. No fitting was performed for this proposal.

## 4. Second candidate: instance-count-sensitive partition features

**Hypothesis.** Current VI is voxel weighted, whereas normalized merge/split risk counts instance events. Joint-entropy normalization also saturates in the documented one-fragment split intervention. A count-sensitive feature could address this mismatch better than another denominator sweep.

One concrete candidate, to be treated as an unvalidated heuristic rather than an established new metric, is **effective counterpart count** on the shared-foreground contingency. For an instance `i` in prediction `P`, define `p(j|i)=n_ij / sum_j n_ij` and `d_i=1/sum_j p(j|i)^2`. Average `d_i−1` equally across instances, repeat in the reverse direction and average the two directions. For an equal-volume split of one shared instance into `k` counterparts this score is `(k−1)/2`, so it does not have the exact VI/H saturation in that toy case. Unequal tiny splits can still be attenuated, and macro weighting may overemphasize spurious small components. This is a hypothesis to test, not a guaranteed severity solution.

**Controls and mechanism checks.** Identity, ID permutation, empty/shared-one-voxel cases, merge/split at multiple levels, unequal fragment volumes, boundary-only changes, missing foreground and spurious tiny objects. Compare with raw VI, existing weighted VI/logN and VI/H, macro conditional entropy, Hungarian disagreement and predicted count variance. Simply writing VI as `H(P|Q)+H(Q|P)` and averaging the sum does not create a different score; a change in weighting/aggregation must be explicit.

**Real-data gate.** First examine the existing validation cohort with fixed definitions and case-grouped evaluation. Add only a small predeclared feature family to the same regularized ranker if mechanism tests support it. A new score should improve fresh-case M+S ranking versus the strongest current controls; a pleasant synthetic severity curve alone is insufficient.

**Compute.** Existing aggregate CSVs do not contain every instance-level contingency. This branch needs CPU rereading of saved predictions, one CT at a time, or sufficiently complete caches. It needs no new GPU inference for the validation cohort. Keep all existing prediction cleanup fixed; changing cleanup at the same time would confound the comparison.

## 5. Third candidate: separate structural ranking from omission safety

**Hypothesis.** A single scalar may be the wrong product objective. Use a structural channel to prioritize partition errors and an omission channel to prevent unanimous misses from being silently accepted.

An inexpensive PENGWIN-specific guard is: for an anatomy expected from the declared scan type, flag the unit if all three members predict no foreground. The existing pair-component counts can express this without consulting GT. This catches only a defined subclass of omissions; it does not detect a shared nonempty but incorrect prediction. More general omission prediction needs independent image/model evidence or supervised failure features, not disagreement alone.

**Important transfer boundary.** This all-empty guard cannot be applied unchanged to RibFrac fracture-lesion detection: 20 of its 80 validation subjects have no GT lesion, and lesion presence is unknown at inference. Correct all-empty RibFrac outputs must not be labeled failures merely to make an omission policy look sensitive. Applicability must follow dataset target semantics; a universal guard requires additional independent evidence.

**Policy.** At a fixed CT review budget, handle a frozen set of omission flags, then allocate the remaining budget using the structural channel. Count a CT once even when multiple bones are flagged. If flags exceed the budget, report that the safety constraint is infeasible at that budget; do not silently exceed the budget or relax the flag after seeing fresh outcomes. Any learned thresholds or budget allocation must be chosen using grouped development data.

**Controls.** Pure partition-v1, pure FA-v2, pure presence, and random under the same reviewed-CT count; the guard alone; and structural ranking without the guard. For a CT score derived from multiple bones, freeze aggregation and check scan-type effects—pelvic CTs contain more annotated bones, so max aggregation creates more opportunities for a high score.

**Metrics.** Retained M+S, `1−F1`, missing-volume risk and complete-omission sensitivity at predeclared 10%/20% CT budgets. Report both bone-macro and case-macro risk summaries where relevant, with paired bootstrap resampling whole CTs. This is a new cost-matched operating analysis, not a relabeling of the existing bone-budget table. Reviewing a case means excluding it from automatic acceptance for this analysis; it does not assume a human corrects it perfectly, and CT counts are not measured reader time.

**Interpretation.** A useful policy would improve omission coverage while preserving a declared acceptable structural-risk trade-off at equal cost. It need not dominate partition-only on every endpoint. Without a predeclared target trade-off, selecting a favorable weighted sum of endpoints after evaluation would be uninformative.

## 6. A minimal sequence with explicit gates

1. **Diagnose on the existing development cohort.** Separate the effect of adding presence, score normalization, prediction size/count and anatomy. Freeze no more than a few plausible candidates; do not search the 100 inspected E3 outcomes for a winning coefficient.
2. **Run a small mechanism gate.** Retain a count-sensitive candidate only if it handles the desired partition interventions and documents its boundary/missing/spurious-object behavior. Preserve failures and exact definitions.
3. **Compare the development procedures with case-grouped validation.** Use nested selection where required. If fusion does not beat the structural reference, retain that negative outcome and consider the two-channel operating goal rather than claiming a new universal scalar.
4. **Freeze one extension before fresh evaluation.** Audit the 110 candidate cases' exposure history, then bind the split, code, features, fitting rule, decision endpoint and strongest control. Evaluate once; report all prespecified contrasts and strata.
5. **Keep E6's current execution intact.** Its active training/inference/evaluation identity is hash bound. Do not change its code/config mid-run to add a new method. An additional scorer is a new output version with a separate protocol. If RibFrac outcomes informed the method, they are exploratory for that extension; a new confirmation set is required.

Fresh PENGWIN confirmation would require at least 330 new member-case inferences for three models on 110 cases; reproducing Stage 2 risk-source sensitivity would bring this to 440. These are counts, not a runtime or storage estimate. Use user-run Slurm with scheduler-assigned GPUs and measured pilot throughput before promising a schedule. Full experiments are not authorized for agent execution here.

## 7. Verification scope and next step

Agent-typed actions were reads of source/config/report files; JSON set arithmetic; metadata-only scan-type lookup through `run_e3.data_inventory`; and CSV membership checks over active score/risk tables. No source volume was decoded, no model was fitted, and no result file was changed. No full command is prepared because the extension is not implemented or frozen. The next supported action is author selection of a bounded candidate family and a development/fresh-confirmation plan, followed by implementation and smoke testing as a separate task.
