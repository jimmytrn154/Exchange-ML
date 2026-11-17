# Annotated Bibliography

Phase 2 · `bibliography_agent` · search executed 2026-11-17 · review form chosen by the author: **scoping review** (ARS has no separate mode; PRISMA-ScR reporting stays with the author — the search log below is written so it can be mapped onto PRISMA-ScR items 6–10).

## Search Strategy

**Databases / indexes:** web search (Claude WebSearch, standard + extended), Crossref REST API (DOI resolution), OpenAlex works search (title resolution for no-DOI venues), arXiv / PMC / PMLR / JMLR full-text pages for primary reading. Semantic Scholar was not queried; OpenAlex served as the deduplication index.
**Keywords (by theme):**
- A. Partition comparison: `variation of information`, `normalized variation of information`, `normalized information distance`, `joint entropy normalisation`, `split merge conditional entropy`, `connectomics segmentation metric`.
- B. Structural / ensemble uncertainty: `topology-aware uncertainty`, `structural uncertainty segmentation`, `structure-wise uncertainty`, `deep ensemble disagreement`, `pairwise Dice failure detection`, `instance segmentation uncertainty merge split`.
- C. Evaluation methodology: `failure detection evaluation AURC`, `risk-coverage`, `selective classification`, `structure size confounder`, `metrics pitfalls image analysis`, `partially labelled evaluation false positives`.
- D. Task / backend: `PENGWIN pelvic fracture segmentation`, `pelvic fracture fragment surgical planning CT`, `nnU-Net`, `STU-Net`, `marginal loss partially supervised`.
**Boolean:** theme terms OR-ed within theme, AND-ed with `segmentation` for B–D; no date filter on A (foundational), 2019–2026 on B–D.
**Date range:** A: none (seminal). B–D: 2019-01-01 to 2026-11-17 (field velocity: rapid).
**Language:** English.
**Document types:** peer-reviewed journal and conference papers; arXiv preprints only when they are the canonical version of a peer-reviewed venue without DOI (ICLR, NeurIPS, MIDL) or the only version of a backend reference (STU-Net); one software repository (backend).
**Last searched:** 2026-11-17.
**Inclusion:** (i) defines or analyses a partition-comparison measure used on segmentations; or (ii) proposes/evaluates an uncertainty or disagreement signal for segmentation failure detection; or (iii) sets evaluation methodology for failure detection or segmentation metrics; or (iv) describes the PENGWIN task, its clinical motivation or the backend components this project uses. Must be resolvable (DOI, or OpenAlex/venue page for no-DOI venues).
**Exclusion:** active-learning acquisition papers (out of scope for this paper; see `__docs__/paper.md` §3.1 for the earlier AL review); RibFrac; unverifiable items (gray zone = FAIL); AI-generated summary sites used only as pointers, never as sources.

**Coverage Distribution Advisory**

```
DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: methodological distribution
- Concentration: computational / methods paper = 26/28 (93%)
- Advisory: This is a coverage-distribution signal, not a defect. The RQ is methodological; two clinical sources (Raniga 2025; Sang 2026 intro) carry the clinical motivation.
- Search response: no expansion; user requested this scope.
```
Time distribution: 2019–2026 = 22/28 (79 %) → advisory triggered; justified by field velocity and by the four foundational partition-comparison sources (2003–2015) being retained. Venue tier and geography: no single value ≥ 70 %.

## PRISMA-style Flow

```
Records identified (total): 31 search queries → ~110 distinct hits screened by title
|-- Web search: ~95
|-- Crossref / OpenAlex resolution: 24 DOIs + 10 title lookups
+-- Project documents (paper.md §3 corpus, re-screened): 9 candidates
Duplicates removed: 14 (preprint vs published; mirror pages)
Records screened (title/abstract): 46
Records excluded: 15 (AL-only 6; RibFrac 1; secondary/AI-summary pages 5; off-topic 3)
Full-text or primary-page assessed: 31
Excluded after assessment (with reasons): 3
   - Funke et al. 2019 TPAMI: exists (DOI verified) but the VOI split/merge claim could not be confirmed from the accessible abstract → not used for that claim
   - Mehrtash et al. 2020 TMI: exists (DOI verified); content read only at abstract level; redundant with Lakshminarayanan + Zenk for the ensemble claim → held in reserve
   - Phitidis et al. 2026 v1 (arXiv): superseded by the published AIM version (kept)
Studies included: 28
```

## Sources (N = 28)

Read-scope key: `full` = full text read; `sections` = named sections read; `abstract` = abstract/landing page only. Trust: `source_verified_against_original: true` only for `full`/`sections`.

### Theme A — Comparing partitions: VI and its normalisations

1. **Meilă, M. (2007). Comparing clusterings—an information based distance. *Journal of Multivariate Analysis, 98*(5), 873–895. https://doi.org/10.1016/j.jmva.2006.11.013**
   - Relevance: defines VI = H(U|V) + H(V|U); proves VI is a metric; discusses bounds (VI ≤ log N; ≤ 2 log K for ≤ K clusters). Our v1 partition term is VI/log N; our v2 is VI/H(U,V).
   - Key findings: VI is a true metric aligned with the lattice of partitions; it is the unique criterion that is both lattice-aligned and convexly additive.
   - Quality: Level VII (theory), seminal; read scope: abstract + Vinh et al. §3.3 account of it.
   - Method weaknesses: not assessed (read scope: abstract).
   - Contribution: the base object for C1 and C2.

2. **Meilă, M. (2003). Comparing clusterings by the variation of information. In *Learning Theory and Kernel Machines* (LNCS 2777, pp. 173–187). Springer. https://doi.org/10.1007/978-3-540-45167-9_14**
   - Relevance: original VI paper; cite with #1 if space allows, otherwise #1 alone.
   - Quality: Level VII; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

3. **Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance. *Journal of Machine Learning Research, 11*, 2837–2854. https://www.jmlr.org/papers/v11/vinh10a.html**
   - Relevance: **the primary source for C1's mechanism.** §3.3: NVI = d_joint = VI/H(U,V) is a metric (Theorem 3); NID = 1 − I/max{H(U),H(V)} is a metric (Theorem 4); Meilă's VI/log N and VI/2 log K normalisations use bounds that "are not as strict as H(U,V)", so "the useful range of these normalized VI variants is narrower than that of d_joint"; footnote 4: if N > RC, H(U,V) ≤ log(RC) ≤ 2 log K. Also: normalisers such as H(U,V) are data-dependent, with "potential side effects on the normalization process" because the normaliser depends on both partitions.
   - Key findings: NVI and NID are both normalised and metric; NID is "generally more preferable" because it uses the [0,1] range better; adjusted-for-chance variants lose the metric property (Theorem 6).
   - Quality: Level VII (theory), JMLR, 2 800+ citations (OpenAlex); read scope: **sections** (§1, §3.2–3.3, footnote 4, abstract).
   - Method weaknesses (theoretical; read: §3): checked: found: assumptions stated, proofs given | checked: none found: empirical scope | not checked: proofs re-derived here. — Data-dependent normalisers may behave differently when one partition is ground truth vs. when both are predictions; distorts comparison when H(U,V) itself varies systematically with the error type. [author-acknowledged, section: 3.3]
   - Contribution: justifies V1 (fixed `VI/H(P,Q)`), names NID as the untested alternative, and supplies the bound that explains the v1 collapse.

4. **Arganda-Carreras, I., Turaga, S. C., Berger, D. R., et al. (2015). Crowdsourcing the creation of image segmentation algorithms for connectomics. *Frontiers in Neuroanatomy, 9*, 142. https://doi.org/10.3389/fnana.2015.00142**
   - Relevance: establishes information-theoretic **split** and **merge** scores for segmentations: V^Info_split = I(S;T)/H(S), V^Info_merge = I(S;T)/H(T), combined as an F-score "closely related to the variation of information" (§2.5); foreground-restricted Rand scoring (§2.4) because border widths are arbitrary.
   - Key findings: VI and Rand measures were "suitable after normalization"; foreground restriction reduced sensitivity to border variation; later border thinning was still needed (§3.4).
   - Quality: Level VI (challenge report), peer-reviewed; read scope: **sections** (2.4, 2.5, 3.4 via PMC).
   - Method weaknesses (empirical; read: §2.4–2.5, 3.4): checked: found: metric definitions, sensitivity analysis | checked: none found: statistical uncertainty on rankings | not checked: participant methods. — Foreground restriction excludes GT border pixels but not predicted ones "to avoid exploitation"; distorts scores when predicted borders are systematically thick. [author-acknowledged, section: 2.4]
   - Contribution: precedent for restricting partition comparison to foreground and for separating split from merge — the design logic behind FA-IPD's two terms.

5. **Nunez-Iglesias, J., Kennedy, R., Parag, T., Shi, J., & Chklovskii, D. B. (2013). Machine learning of hierarchical clustering to segment 2D and 3D images. *PLoS ONE, 8*(8), e71715. https://doi.org/10.1371/journal.pone.0071715**
   - Relevance: advocates VI as the accuracy measure for 3D EM instance segmentation; shows VI used at voxel scale in practice.
   - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

6. **Kirillov, A., He, K., Girshick, R., Rother, C., & Dollár, P. (2019). Panoptic segmentation. In *CVPR 2019* (pp. 9396–9405). https://doi.org/10.1109/CVPR.2019.00963**
   - Relevance: the IoU-threshold matching convention (≥ 0.5 there; ≥ 0.1 in our frozen contract) that condition C3(d) examines.
   - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

### Theme B — Uncertainty, ensembles and structural failure in segmentation

7. **Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In *NeurIPS 30*. arXiv:1612.01474**
   - Relevance: the ensemble construction (M = 3, independent seeds) we use.
   - Quality: Level VI, canonical; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

8. **Jungo, A., & Reyes, M. (2019). Assessing reliability and challenges of uncertainty estimations for medical image segmentation. In *MICCAI 2019* (LNCS 11765, pp. 48–56). https://doi.org/10.1007/978-3-030-32245-8_6**
   - Relevance: voxel-wise uncertainty methods perform similarly and are miscalibrated at subject level → motivates subject/structure-level signals.
   - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

9. **Roy, A. G., Conjeti, S., Navab, N., & Wachinger, C. (2019). Bayesian QuickNAT: Model uncertainty in deep whole-brain segmentation for structure-wise quality control. *NeuroImage, 195*, 11–22. https://doi.org/10.1016/j.neuroimage.2019.03.042**
   - Relevance: structure-wise uncertainty from MC samples; IoU across samples as a Dice proxy — the semantic-level analogue of our presence term.
   - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

10. **Mehrtash, A., Wells, W. M., Tempany, C. M., Abolmaesumi, P., & Kapur, T. (2020). Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. *IEEE Transactions on Medical Imaging, 39*(12), 3868–3878. https://doi.org/10.1109/TMI.2020.3006437**
    - Relevance: ensembles improve calibration of segmentation confidence; reserve citation.
    - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

11. **Zenk, M., Zimmerer, D., Isensee, F., et al. (2025). Comparative benchmarking of failure detection methods in medical image segmentation: Unveiling the role of confidence aggregation. *Medical Image Analysis, 101*, 103392. https://doi.org/10.1016/j.media.2024.103392**
    - Relevance: **closest prior art for our baselines.** Finds "superior performance of the pairwise Dice score" as "a simple and robust baseline for failure detection", stresses "pixel confidence aggregation", and advocates "risk-coverage analysis as a holistic evaluation approach"; five public 3D datasets.
    - Key findings: pairwise Dice between ensemble members beats most voxel-confidence aggregates for case-level failure detection; aggregation choice matters as much as the uncertainty method.
    - Quality: Level VI (benchmark), MedIA; read scope: abstract (arXiv landing). Method weaknesses: not assessed (read scope: abstract).
    - Contribution: our pairwise-Dice and Jaccard rows are this baseline; our question is whether anything beyond presence disagreement adds signal for *instance* failures, which this benchmark (semantic targets) does not test.

12. **Kahl, K.-C., Lüth, C. T., Zenk, M., Maier-Hein, K., & Jaeger, P. F. (2024). ValUES: A framework for systematic validation of uncertainty estimation in semantic segmentation. In *ICLR 2024*. arXiv:2401.08501**
    - Relevance: decomposes an uncertainty method into backbone / prediction model / uncertainty measure / **aggregation** (C0–C3); failure detection as one of five downstream tests.
    - Quality: Level VI, ICLR oral; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).
    - Contribution: FA-IPD is an aggregation choice (C3) over instance maps rather than voxels — positions the paper in ValUES' terms.

13. **Geißler, K., Hirsch, J. G., Heldmann, S., & Meine, H. (2024). Structure size as confounder in uncertainty based segmentation quality prediction. In *MIDL 2024* (PMLR 250, pp. 504–519). https://proceedings.mlr.press/v250/geissler24a.html**
    - Relevance: structure size confounds uncertainty–quality correlations; remedy: partial correlation controlling for size; measures studied include mean pairwise Dice and a volume-variation coefficient; ensembles, MC dropout, TTA.
    - Quality: Level VI; read scope: abstract (PMLR page). Method weaknesses: not assessed (read scope: abstract).
    - Contribution: direct precedent for our confounder-controlled Spearman (fragment count, bone volume, min-fragment volume).

14. **Gupta, S., Zhang, Y., Hu, X., Prasanna, P., & Chen, C. (2023). Topology-aware uncertainty for image segmentation. In *NeurIPS 36*. arXiv:2306.05671**
    - Relevance: uncertainty over topological structures (discrete Morse theory) for curvilinear objects; inter- vs intra-structural uncertainty.
    - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).
    - Contribution: establishes "structure-wise" uncertainty as a field; differs from ours (curvilinear, DMT; not variable-cardinality instances).

15. **Yang, B., Zhang, X., Zhang, H., Li, S., Higashita, R., & Liu, J. (2025). Structural uncertainty estimation for medical image segmentation. *Medical Image Analysis, 103*, 103602. https://doi.org/10.1016/j.media.2025.103602**
    - Relevance: "structural uncertainty" via active shape models (SU-ASM); shape-prior based, single-object.
    - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).
    - Contribution: shows the term "structural uncertainty" is in use for shape-level uncertainty; our usage (partition of foreground into instances) must be distinguished explicitly.

16. **Berger, A. H., Lux, L., Weers, A., Menten, M., Rueckert, D., & Paetzold, J. C. (2025). Pitfalls of topology-aware image segmentation. In *IPMI 2025* (LNCS, pp. 297–312). https://doi.org/10.1007/978-3-031-96628-6_20**
    - Relevance: three pitfalls — connectivity choice, topological artifacts in ground truth, inappropriate metrics — and their effect on method ranking.
    - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).
    - Contribution: supports C3 framing (evaluation choices change rankings) and our frozen 26-connectivity rule. A secondary page claims the authors criticise VOI for mixing volumetric and topological information — **not verified against the primary text; do not cite that specific claim.**

### Theme C — Evaluating failure detection and segmentation metrics

17. **Jaeger, P. F., Lüth, C. T., Klein, L., & Bungert, T. J. (2023). A call to reflect on evaluation practices for failure detection in image classification. In *ICLR 2023* (oral). arXiv:2211.15259**
    - Relevance: AURC as the primary failure-detection metric; failure detection distinct from calibration; softmax-response baseline often best.
    - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

18. **Geifman, Y., Uziel, G., & El-Yaniv, R. (2019). Bias-reduced uncertainty estimation for deep neural classifiers. In *ICLR 2019*. arXiv:1805.08206**
    - Relevance: AURC / E-AURC origin in selective prediction.
    - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

19. **Maier-Hein, L., Reinke, A., Godau, P., et al. (2024). Metrics reloaded: Recommendations for image analysis validation. *Nature Methods, 21*(2), 195–212. https://doi.org/10.1038/s41592-023-02151-z**
    - Relevance: problem-fingerprint-driven metric selection; covers instance segmentation; warns of pitfalls incl. prevalence dependence.
    - Quality: Level VII (consensus, Delphi), high authority; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

20. **Reinke, A., Tizabi, M. D., Baumgartner, M., et al. (2024). Understanding metric-related pitfalls in image analysis validation. *Nature Methods, 21*(2), 182–194. https://doi.org/10.1038/s41592-023-02150-0**
    - Relevance: taxonomy of metric pitfalls; relevant to C3(b) (prevalence) and C3(d) (matching thresholds).
    - Quality: Level VII; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

21. **Phitidis, J., Smithard, A. Q., Whiteley, W. N., Wardlaw, J. M., Bernabeu, M. O., & Valdés Hernández, M. (2026). Comparative evaluation of training strategies using partially labelled datasets for segmentation of white matter hyperintensities and stroke lesions in FLAIR MRI. *Artificial Intelligence in Medicine, 181*, 103507. https://doi.org/10.1016/j.artmed.2026.103507**
    - Relevance: states in Limitations that "the inclusion of partially labelled data within the evaluation splits … inherently limits our capacity to systematically quantify false-positive predictions for unannotated classes" — the generic form of our D4 / C3(c).
    - Quality: Level VI; read scope: **sections** (2.2, 3.3, 4.1, 5 Limitations via arXiv HTML).
    - Method weaknesses (empirical; read: §2.2, 3.3, 4.1, 5): checked: found: limitations stated, annotation-protocol variation acknowledged | checked: none found: per-class FP quantification | not checked: training details. — Partially labelled test scans; distorts FP rates for unannotated classes. [author-acknowledged, section: 5 Limitations]

### Theme D — Task, clinical context and backend

22. **Sang, Y., et al. (2026). Benchmark of segmentation techniques for pelvic fracture in CT and X-ray: Summary of the PENGWIN 2024 challenge. *IEEE Transactions on Medical Imaging, 45*(5), 2212–2228. https://doi.org/10.1109/TMI.2025.3650126** (preprint arXiv:2504.02382)
    - Relevance: lineage of our dataset (2024 edition: 150 CTs from six centres; fragments per case 3–12, 5.7 ± 1.4; labels sacrum / left hip / right hip with size-ordered fragment IDs; metrics IoU-F, HD95-F, ASSD-F; matching "each ground-truth fragment was matched to the predicted fragment with the highest IoU"). Clinical motivation: "Automated pelvic fracture segmentation is a crucial step in trauma diagnosis and surgical planning"; "precise fracture segmentation is a prerequisite for fragment-wise reduction planning".
    - Quality: Level VI (challenge report), TMI; read scope: **sections** (Introduction, Dataset, Evaluation via arXiv HTML).
    - Method weaknesses (challenge report; read: intro, dataset, evaluation): checked: found: dataset description, metric definitions | checked: none found: confidence intervals on team rankings | not checked: team methods. — Highest-IoU matching without a minimum threshold; distorts fragment-wise scores when a predicted fragment is matched to several GT fragments (merge) [reader-inferred].
    - Note: our data is the **2026** edition (340 labelled cases; `__docs__/data_description.md`); no peer-reviewed description of the 2026 edition was found. Cite #22 for lineage and the challenge website for the 2026 data.

23. **Liu, Y., Yibulayimu, S., Sang, Y., Zhu, G., Wang, Y., Zhao, C., & Wu, X. (2023). Pelvic fracture segmentation using a multi-scale distance-weighted neural network. In *MICCAI 2023* (LNCS 14223, pp. 312–321). https://doi.org/10.1007/978-3-031-43996-4_30**
    - Relevance: the FracSegNet lineage and dataset host behind PENGWIN.
    - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

24. **Raniga, S., Pal, D., Mehta, N., et al. (2025). High-energy pelvic ring injuries: A comprehensive imaging review. *RadioGraphics, 45*. https://doi.org/10.1148/rg.250027**
    - Relevance: clinical hook — CT delineation of fragment rotation, comminution and displacement guides classification (Tile / Young–Burgess) and reduction planning; displacement thresholds drive surgical indication.
    - Quality: Level VII (narrative review), RSNA; read scope: abstract/snippets. Method weaknesses: not assessed (read scope: abstract).
    - Note: fragment **count** is not itself a surgical cutoff in the sources found; the paper's clinical sentence must be about fragment delineation/displacement, not count.

25. **Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods, 18*(2), 203–211. https://doi.org/10.1038/s41592-020-01008-z**
    - Relevance: backend framework (v2.5.1 pinned).
    - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

26. **Huang, Z., Wang, H., Deng, Z., Ye, J., Su, Y., et al. (2023). STU-Net: Scalable and transferable medical image segmentation models empowered by large-scale supervised pre-training. arXiv:2304.06716**
    - Relevance: architecture family used by the ABBC backend (STU-Net base, trained from scratch here).
    - Quality: Level VI, **preprint** (contamination signal: preprint_post_llm_inflection = false, year 2023); read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

27. **Shi, G., Xiao, L., Chen, Y., & Zhou, S. K. (2021). Marginal loss and exclusion loss for partially supervised multi-organ segmentation. *Medical Image Analysis, 70*, 101979. https://doi.org/10.1016/j.media.2021.101979**
    - Relevance: the marginal (partial-label) loss the Stage A trainer uses; defect D3 was the silent absence of its case map.
    - Quality: Level VI; read scope: abstract. Method weaknesses: not assessed (read scope: abstract).

28. **RURUGURU. (2026). *pengwin2026-task1-abbc* [software]. GitHub. Commit `ccc625f`; project patch in `patches/pengwin2026-task1-abbc_on_ccc625f.patch`.**
    - Relevance: the two-stage anatomy/fragment backend; our six defect fixes apply to it.
    - Quality: Tier 3 grey (software). Verified only as a local checkout; not independently verified online in this run.

## Search Limitations

- Single-run web search plus two bibliographic APIs; no Scopus/WoS; no citation chasing beyond one hop. A PRISMA-ScR-compliant scoping review would require a second screener and a registered protocol — those remain with the author.
- Most sources read at abstract level (`abstract` scope); method-weakness entries are therefore mostly "not assessed", as the rules require.
- No source was found that measures ensemble disagreement over **instance partitions** against merge/split error in 3D medical CT; the absence claim is bounded to the queries listed above (themes B and D, 2023–2026).
- The PENGWIN 2026 edition has no peer-reviewed description found; dataset facts come from the project's own audit (`__docs__/data_description.md`).
