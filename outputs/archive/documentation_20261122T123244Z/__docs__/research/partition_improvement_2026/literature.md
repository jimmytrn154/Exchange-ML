# Literature evidence for improving partition-sensitive quality control

**Checked 22 November 2026. Scope:** targeted primary-source extraction for an improvement proposal, not implementation or a new empirical result. The papers below motivate candidate mechanisms; none establishes that a modified score will beat the frozen PENGWIN partition-only control. No existing config, scorer, checkpoint or experiment output was changed.

## 1. Learned aggregation can help, but small-data failure is a real counterexample

**Zenk et al. (2025), Medical Image Analysis 101, 103392.** *Comparative benchmarking of failure detection methods in medical image segmentation: Unveiling the role of confidence aggregation.* [DOI](https://doi.org/10.1016/j.media.2024.103392), [published full text, DKFZ](https://inrepo02.dkfz.de/record/294922/files/1-s2.0-S1361841524003177-main.pdf?version=1).

Sections 4.4.2–4.4.3 distinguish untrained aggregation, regression forests and image/mask quality regression. A simple forest uses confidence in foreground/background/boundary, foreground fraction and component count; targets are DSC values from cross-validation predictions. Thus deployment needs no test GT, but fitting needs labeled development cases. Sections 5.2–6.2 find improvements over mean entropy on several datasets, yet a severe prostate-set failure with only 21/5 training/validation examples. Pairwise Dice remains their strongest overall baseline. Their practitioner advice explicitly distinguishes OOD detection from failure detection and recommends matching the risk function to the application.

**Relevance, not a paper result:** use this as a reason to test a very small, regularized, endpoint-specific fusion of existing partition/presence features, with case-level out-of-fold development. It is evidence against assuming that a flexible random forest or high-dimensional feature bank will work with 40 development CTs. The paper predicts DSC; substituting merge/split or omission targets is a new adaptation.

**Locators:** printed pp. 7–8, §§4.4.2–4.4.3; pp. 8–10, §5.2/Fig. 3; pp. 11–13, §6.1–6.2. PDF text read; numerical figure results are not used to estimate an expected gain for Exchange-ML.

## 2. Spatial structure and meta-aggregation: useful ideas, not automatic risk calibration

**Guarino et al. (2026), CVPR.** *Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance.* [Official CVF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Guarino_Better_than_Average_Spatially-Aware_Aggregation_of_Segmentation_Uncertainty_Improves_Downstream_CVPR_2026_paper.pdf), [author HTML full text](https://arxiv.org/html/2603.29941v1).

Sections 3.1–3.3 compare scalar summaries and construct spatial mass ratios using local autocorrelation, edges and entropy. Section 3.4 represents each uncertainty map by several aggregation features, fits a GMM on a separate known-in-distribution fitting set, chooses mixture complexity using BIC, and uses feature negative log-likelihood as the final score. Fitting is density estimation, not supervised regression of the clinical failure endpoint. Appendix H explicitly warns about low-data/high-dimensional GMM fitting. Transfer of any chosen aggregator across datasets is not guaranteed.

**Relevance, not a paper result:** spatial patterns could help separate small localized partition defects from diffuse foreground uncertainty. However, averaging foreground-probability entropy differently cannot detect a pure partition change with identical foreground. A partition-boundary/co-membership disagreement map must first be defined without using arbitrary label values. A 3D extension also needs physical-spacing-aware neighborhoods and frozen development choices. GMM is a secondary control, not the preferred first repair under 40-case development.

**Locators/access:** author full text §§3.1–3.4, Appendix B.2/G/H; official CVF search-indexed version verifies publication. No quantitative gain is imported into this proposal.

## 3. Instance correspondence separates detection consistency from shape consistency

**Siddiqui, Starke and Steinbach (2024), WACV.** *Uncertainty Estimation in Instance Segmentation With Star-Convex Shapes.* [Official CVF paper](https://openaccess.thecvf.com/content/WACV2024/papers/Siddiqui_Uncertainty_Estimation_in_Instance_Segmentation_With_Star-Convex_Shapes_WACV_2024_paper.pdf), [author full text](https://arxiv.org/html/2309.10513v1), [official proceedings index](https://openaccess.thecvf.com/WACV2024).

Their mask-based Pixel Approach groups predicted objects across MC-dropout/ensemble passes using IoU-based sequential clustering. Spatial certainty averages overlap with the median cluster mask; fractional certainty measures how consistently an object is detected across passes; hybrid certainty multiplies these. The separate Radial Approach instead requires StarDist object probabilities and radial distances before NMS. The paper evaluates instance confidence calibration, not normalized merge/split-count ranking for fracture CT.

**Relevance, not a paper result:** mask-based correspondence can be adapted to the three saved member maps, without retraining a StarDist model. For fracture partitions, preserve one-to-many/many-to-one overlap relations rather than forcing every relation into a single matched cluster, which could hide merge/split evidence. Per-object detection and fragmentation features, with explicit unmatched objects, are a more direct candidate than raw voxel-ID variance. Do not assume fragments are star-convex or that three members provide the same uncertainty precision as many MC samples.

**Locators/access:** author §§3.1.1, 3.2.1, 3.3–3.4; official PDF indexed section 3.3. Direct CVF fetch returned HTTP403, so technical extraction uses the author version plus indexed official text. Proceedings metadata are verified; differing pagination between versions is not resolved here.

## 4. Structure-wise QC supports separate channels, not certainty of correctness

**Roy et al. (2019), NeuroImage 195, 11–22.** *Bayesian QuickNAT: Model uncertainty in deep whole-brain segmentation for structure-wise quality control.* [DOI](https://doi.org/10.1016/j.neuroimage.2019.03.042), [PubMed](https://pubmed.ncbi.nlm.nih.gov/30926511/), [author full text](https://arxiv.org/pdf/1811.09800).

The paper uses MC-dropout predictions and four structure-wise summaries: volume coefficient of variation, average pairwise Dice, intersection/union across all samples, and mean uncertainty within the predicted structure. These output-based measures do not need GT for a new scan; GT is used to assess their association with segmentation quality. Anatomical identities are fixed brain labels, whereas fracture-fragment identities are arbitrary and need correspondence.

**Relevance:** predicted-volume consistency and structure-wise uncertainty can complement partition features. Much of the per-bone/overlap logic is already present in Exchange-ML and is not a new contribution by itself. Any measure based only on repeated predictions remains unable to distinguish unanimously correct from unanimously wrong masks when its inputs are identical. Three members missing the same structure cannot be rescued merely by changing the normalization of their disagreement.

**Locators:** author PDF §2.3, especially Eqs. 6–9, pp. 3–4; publication metadata verified separately by PubMed. Correlations in the paper are not evidence of omission-safe fracture review.

## 5. Common errors require evidence beyond agreement: RCA supplies one example

**Valindria et al. (2017), IEEE Transactions on Medical Imaging 36(8), 1597–1606.** *Reverse Classification Accuracy: Predicting Segmentation Performance in the Absence of Ground Truth.* [DOI](https://doi.org/10.1109/TMI.2017.2665165), [publisher record](https://ieeexplore.ieee.org/document/7902121), [accepted author manuscript](https://arxiv.org/pdf/1702.03407).

RCA trains a reverse classifier on each new image using its predicted segmentation as pseudo-GT. The classifier is evaluated on reference images with known labels; best reference performance is used as a proxy for the new image's unknown quality. It does not require GT for that new image, but it does require a labeled reference bank and extra per-case learning or registration. Thus “GT-free at deployment” does not mean “no labeled data anywhere.”

**Relevance:** the mechanism introduces image/reference evidence independent of ensemble agreement. It is a proof of an architectural alternative, not a drop-in partition metric. Correspondence of arbitrary fracture fragments and per-case computational cost make full RCA a poor near-term default for this project. A restricted expected-anatomy sanity check might be cheaper, but is a separate hypothesis and needs validation of its field-of-view assumptions.

**Locators:** author manuscript pp. 2–4, §II and definition of proxy performance. No claim is made that RCA necessarily catches every shared mistake.

## 6. Metric design should respect object sizes and the operational endpoint

**Reinke et al. (2024), Nature Methods.** *Understanding metric-related pitfalls in image analysis validation.* [Publisher/DOI](https://www.nature.com/articles/s41592-023-02150-0), [author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC11181963/).

The taxonomy distinguishes problems caused by object size, shape, topology, output type and inappropriate aggregation. Section P2.2 discusses small or variable-sized structures and touching/disconnected objects; P3 addresses application and hierarchical aggregation. It does not prescribe a particular new partition uncertainty score or show that one metric is universally preferable.

**Relevance:** voxel-mass-weighted information disagreement and count-normalized merge/split risk answer different questions. An object-balanced overlap/branching summary is therefore a defensible candidate to test, not a literature-proven improvement. Compare count-sensitive, voxel-sensitive and omission-sensitive outcomes separately, preserve case clustering, and avoid choosing an endpoint only because a candidate wins on it. Keep the original endpoint as a control even when adding exploratory outcomes.

**Locators/access:** publisher full text, P2.2/P2.4/P3 and related supplementary-note descriptions. Only textual design principles are extracted; no quantitative performance claim is used.

## Candidate translation, with evidence boundaries

| Candidate for this project | What the literature supports | What is still our hypothesis / required evidence |
|---|---|---|
| Very small calibrated structural-risk channel using partition plus optional presence/volume/count features | Quality regression and learned aggregation can use prediction-only inputs at deployment | Fit only on labeled development predictions; case-wise out-of-fold validation; beat partition-v1/v2, not just entropy; independent untouched confirmation after design |
| Per-object overlap graph and count-sensitive aggregation | Existing instance-QC methods associate sampled masks and separate detection/shape consistency | A many-to-many graph better preserves fracture merge/split evidence than forced matching; requires ID-invariant toy cases and tests with unchanged foreground/small fragments |
| Local partition-boundary/co-membership disagreement | Spatial aggregation can capture patterns lost by global averaging | Need a valid partition map; foreground entropy alone is insufficient; 3D physical neighborhoods and useful transfer remain untested |
| Separate omission/QC channel with expected anatomy or independent image evidence | Reference-based/image-based quality estimation is possible without new-case GT | An all-empty flag is only sensible if scan metadata guarantees expected anatomy. It must not be copied to RibFrac, where fracture-free subjects are valid negatives |

For a low-cost first experiment, the literature favors preserving the strongest partition-only controls and checking a small number of endpoint-aligned features before training another segmentation network. High-capacity RF/GMM/quality CNNs are higher-risk choices with 40 development cases. Hyperparameter tuning against the already inspected 100-case held-out outcomes would make them exploratory; it would not repair the original confirmation claim. Existing E6 methods and writer identities should remain unchanged during training.

**Evidence versus proposal:** source method descriptions above are verified. All translations to PENGWIN fracture partitions are proposed adaptations, not measured gains, adopted protocol amendments or novelty claims. The algebraic impossibility of recovering correctness from indistinguishable prediction-only inputs is a reasoning constraint, not a reported clinical result.

**Agent-typed work:** primary-source searches and targeted method-section reading; creation of this source ledger only. **User-typed full runs prepared/executed:** none. No model fitting, new held-out evaluation, bootstrap or source-data modification was performed.
