# Tiến độ và kết quả

Cập nhật ngày **22/11/2026** từ artifacts hiện có. Đây là báo cáo tổng hợp duy nhất cho các stage đã hoàn tất; protocol ở [exps.md](exps.md), công việc còn lại ở [plan.md](plan.md).

## Trạng thái tổng thể

| Hạng mục | Trạng thái | Kết quả / giới hạn |
|---|---|---|
| Stage 0–3: split, instance audit, backend, ba members | Hoàn tất phần triển khai và artifacts | Giữ splits, patch F1–F9, checkpoints và manifests; không tự thay đổi sign-off G2/H1 lịch sử |
| H1 v1 | Hoàn tất, không đạt giả thuyết ưu thế | Không có lợi thế FA-IPD theo ba tiêu chí lịch sử; không dùng làm kết quả dương |
| E1 | Hoàn tất: 40 CT / 80 bones | Phân rã presence/partition và chẩn đoán mẫu số |
| E2 | Hoàn tất: 20 CT / 40 bones | 32/32 phantom assertions; 452 perturbations hợp lệ |
| E3 | Hoàn tất: 100 CT / 200 bones / 400 inference records | Đạt criterion pooled so với presence/entropy; partition-only mạnh hơn |
| E4–E5 và E5 supplementary | Hoàn tất | Có sensitivity, blindspots, qualitative, runtime; chưa chứng minh review utility |
| Partition v3 | Hoàn tất development: 40 CT / 80 bones | Ridge fusion kém P1; G đơn lẻ gần ngang P1 |
| Partition v4 | Hoàn tất development: 40 CT / 80 bones | P1+G chưa cải thiện có ý nghĩa; không phải fresh confirmation |
| E6 RibFrac fast | Chưa hoàn tất | Handoff/cache đã có artifacts; training đang cập nhật log; chưa có external evaluation được audit |
| Manuscript | Chưa hoàn tất | Có bản nháp Results/Discussion trong archive; chưa submission-ready |

## Stage 0–3 và H1 lịch sử

- Split PENGWIN, instance metrics, sửa backend ABBC và ba members seeds `31001–31003` đã cung cấp đầu vào cho E1–E5. Giữ 26-connectivity, GT semantics và các prediction/probability files đã kiểm tra.
- Stage 2 sau sửa backend: instance F1 **0,8502**, binary Dice **0,7303** trên 40 validation CT theo `outputs/stage2_backend/validation_summary.json`. Đây là kết quả backend lịch sử, không thay bảng per-bone E3.
- H1 v1: báo cáo đề nghị **FAIL**; artifact `outputs/stage3_h1/h1_summary.json` còn ghi `PENDING_SCIENTIFIC_ASSESSMENT`. Việc gộp tài liệu không tự thay đổi quyết định gateway.
- Các kết quả âm tính, patch, manifests, logs và checkpoints được giữ để đối chiếu; không xóa như file thừa.

<a id="e1"></a>

## E1: tín hiệu partition có thật, nhưng phải phân biệt với whole-case lịch sử

E1 per-bone dùng cùng cleanup và units với E3: tính score không dùng GT voxels, rồi đánh giá risk từng model trước khi mean ba member. Presence là foreground Jaccard disagreement. Weighted partition terms là `w·VI/log(N)` và `w·VI/H`, với `N` là số voxel shared foreground, `H` là joint entropy, `w` là shared/union foreground fraction. Mean ba cặp member được lấy trước khi tóm tắt qua bone units.

| Quantity, pooled 80 bones | Median | Maximum |
|---|---:|---:|
| Raw VI, nats | 0,014834 | 0,601793 |
| Weighted partition v1 | 0,001164 | 0,047092 |
| Weighted partition v2 | 0,032116 | 0,988552 |
| Partition share trong FA-v1 | 6,9581% | 76,8585% |
| Partition share trong FA-v2 | 58,8711% | 98,8583% |

Raw mean-pair VI ≤1e−12 ở **27/80 bones (33,75%)**. Vì nhiều bones khác vẫn có VI dương, việc cả ensemble hoàn toàn đồng ý về partition không giải thích toàn bộ dữ liệu. Chuẩn hóa joint entropy làm contribution tăng; việc tăng magnitude không tự chứng minh ranking tốt nhất.

Các số **“share median khoảng 2%, max contribution <0,02, rank correlation 0,997” thuộc whole-case analysis lịch sử**. Không thay chúng vào current per-bone E1: đơn vị, scope và AURC convention đã khác. Các source chính cho E1 hiện tại là `tables/decomposition.csv` và `validation/*`, không phải bảng H1 lịch sử. [Evidence: E1_decomposition, E1_validation_contrasts](../outputs/stage3_h1_v2/reporting/e1_e5_evidence.json).

<a id="e2"></a>

## E2: thành phần phản ứng đúng trong perturbations, có giới hạn severity

**32/32 phantom assertions đạt**. Các eligible responses gồm identity 40, ID permutation 40, merge 45, split 120, missing 87 và boundary 120. Có 75 merge rows ineligible vì không đủ fragments; không loại im lặng khỏi denominator.

- Identity và ID permutation giữ presence/VI/Q2 bằng 0.
- Eligible merge/split giữ presence bằng 0 và tạo VI/Q2 dương.
- Missing/boundary có presence dương, VI/Q2 bằng 0 trong các can thiệp đã định.
- Cả 120 boundary targets đạt tolerance 0,005; không suy rộng thành robustness với mọi kiểu boundary error.

Joint normalization có saturation: **11 bones với một GT fragment có Q2=1 ở cả ba mức split**. Nó phát hiện thay đổi partition nhưng không đếm mức độ lỗi split một cách phổ quát.

Merge cohorts thay đổi theo eligibility: 29/12/4 bones ở mức 1/2/3. Median trên các nhóm khác nhau giảm không phải within-bone response. Đối chiếu matched cohorts:

| Cùng bones ở mọi mức trong hàng | Q2 ở mức 1 | Mức 2 | Mức 3 |
|---|---:|---:|---:|
| 12 bones đủ mức 1/2 | 0,015032 | 0,033021 | — |
| 4 bones đủ mức 1/2/3 | 0,005722 | 0,017858 | 0,032355 |

[Hình matched-cohort mới cho báo cáo](../outputs/stage3_h1_v2/reporting/e2_matched_merge.pdf) giữ cùng bones qua mọi severity trong mỗi panel, hiển thị trajectories và median, không có CI hay claim clinical severity. Hình được dựng từ CSV đã hash-verify; các figures gốc của E2 không bị thay thế.

E2c tái dựng **240 E1 contingencies**: 154 positive-VI + 86 zero-VI pairs. Nhân mọi union counts, kể cả foreground mismatch, với `r={1,2,4,8,16}` giữ VI, H, presence, weight và Q2 bất biến; Q1 đi theo `VI/log(rN)`. Residuals ghi trong summary bằng 0. Có 775 positive-VI/phantom replication rows và 430 zero-VI rows. Đây là phép thử đếm chính xác, không phải CT resampling hoặc chứng minh invariance theo độ phân giải ảnh. [Evidence: E2 summaries và derived/matched_merge_q_joint](../outputs/stage3_h1_v2/reporting/e1_e5_evidence.json).

<a id="e3"></a>

## E3: tiêu chí chính đạt, không phải thắng mọi baseline

Primary risk là mean per-model normalized merge+split risk của ba member. Mỗi bone có cùng trọng số trong pooled AURC; pelvic đóng góp ba bones/case, femur một. Case-cluster bootstrap mang theo các bones và multiplicity, phân tầng scan type. AURC dùng expected ordering trong exact ties và trapezoid trên coverage `1/n…1`, không thêm coverage=0.

**AURC thấp hơn tốt hơn.** Tất cả hàng dưới dùng cùng current per-bone convention.

| Score | Validation M+S | Held-out M+S | Held-out 1−F1 |
|---|---:|---:|---:|
| Mean entropy | 0,6570 | 0,6358 | 0,1757 |
| P95 entropy | 0,6767 | 0,6381 | 0,1784 |
| Pairwise Dice disagreement | 0,5442 | 0,4943 | 0,1611 |
| Presence | 0,5442 | 0,4942 | 0,1611 |
| Union-NVI | 0,4342 | 0,3881 | 0,1534 |
| Count variance | 0,5181 | 0,4704 | 0,1523 |
| Hungarian | 0,3933 | 0,3883 | 0,1331 |
| FA-v1 | 0,4906 | 0,4472 | 0,1557 |
| Partition-v1 | 0,2706 | **0,2720** | 0,1446 |
| Partition-v2 | 0,2891 | 0,2978 | 0,1424 |
| FA-v2 | 0,3318 | 0,3311 | 0,1453 |
| Unweighted joint Q | 0,2890 | 0,2987 | 0,1414 |

Held-out contrasts `Δ=AURC(FA-v2)−AURC(baseline)`:

| Risk source | Baseline | ΔAURC | 95% case-bootstrap CI |
|---|---|---:|---|
| Member-mean | Presence | −0,16313 | [−0,21184; −0,11682] |
| Member-mean | Mean entropy | −0,30470 | [−0,37159; −0,23683] |
| Stage 2 | Presence | −0,12760 | [−0,18586; −0,07138] |
| Stage 2 | Mean entropy | −0,28142 | [−0,35085; −0,21028] |

Cả hai primary member-mean contrasts đạt effect floor −0,02 và CI dưới 0. Stage 2 cũng ủng hộ cùng hai contrasts, nhưng cùng backend family nên không phải cross-backend validation. Femur-only member-mean CIs so với presence/entropy cắt 0; không khẳng định đã xác nhận riêng ở cả hai scan types.

Partition-only tốt hơn FA-v2 trong pooled held-out M+S: FA-v2−partition-v1 **+0,05914 [0,03921; 0,08032]**, FA-v2−partition-v2 **+0,03333 [0,01535; 0,05265]**. Đây là secondary contrasts, không chuyển primary sang hàng thắng sau khi xem held-out. Hungarian có point AURC 1−F1 thấp nhất trong bảng, nên không gọi FA-v2 là detector tốt nhất cho mọi loại failure. [Evidence: E3_main_table và E3_structural_contrasts](../outputs/stage3_h1_v2/reporting/e1_e5_evidence.json).

<a id="e4"></a>

## E4: bốn điều kiện đánh giá phải được tách riêng

### Normalization × annotation scope

E4a là **case-level 2×2**. Trong mỗi model, risk bằng tổng merge+split trên anatomy được annotate chia tổng GT instance count, rồi mean ba members. Không so absolute AURC E4 với E3 per-bone để quy phần chênh cho scope; cả unit weighting và risk aggregation khác nhau.

Pooled held-out case-level contrasts:

| Controlled change | Member-mean ΔAURC [95% CI] | Stage 2 ΔAURC [95% CI] |
|---|---|---|
| Joint H thay log N, all scope | −0,03036 [−0,07766; 0,01558] | −0,03756 [−0,08699; 0,01119] |
| Joint H thay log N, annotated scope | **−0,07030 [−0,13079; −0,01190]** | **−0,06979 [−0,13052; −0,01147]** |
| Annotated thay all scope, log N | +0,05529 [−0,02067; 0,13445] | +0,02081 [−0,06900; 0,11039] |
| Annotated thay all scope, joint H | +0,01535 [−0,05856; 0,09291] | −0,01141 [−0,09585; 0,06838] |

Joint normalization có lợi trong pooled annotated-scope case ranking; **annotation restriction tự nó chưa cho thấy AURC improvement**. Các CIs của annotated normalization ở cả hai scan-type strata, cho cả hai risk sources, cũng cắt 0. Các CI trên là marginal secondary intervals, không phải bốn primary tests mới.

### Disagreement ngoài annotation scope

Quantity sau là mean qua cases của mean ba pair ratios `|D∩O|/|D|`, với `D` là foreground symmetric difference và `O` là vùng prediction gán anatomy ngoài scope:

| Split | Pelvic mean | Femur mean |
|---|---:|---:|
| Validation | 31,62% | 87,07% |
| Held-out | 29,66% | 86,55% |

Held-out femur median của case means là 93,07%. Đây là **predicted anatomy attribution**, không phải GT xác nhận mọi voxel đó là bone chưa annotate. Mean pooled case ratios là 58,11%, trong khi ratio của summed voxel counts là 79,69%; hai cách aggregation trả lời hai câu hỏi khác nhau. Disagreement ngoài GT foreground được lưu riêng vì còn chứa boundary errors và false positives. [Evidence: derived/annotation_scope_case_summary; scope pair/outside-GT CSVs](../outputs/stage3_h1_v2/reporting/e1_e5_evidence.json).

### Binary prevalence khác ưu tiên lỗi nặng

Any-M+S prevalence held-out là **138/200=69%** ở member-mean và **125/200=62,5%** ở Stage 2, cho binary enrichment ceiling 1,4493 và 1,6000. Femur member-mean có **48/50=96%**, ceiling chỉ 1,0417. Không dùng “92,5% ca” của historical whole-case analysis để mô tả các denominator per-bone này.

Enrichment trên pooled held-out member-mean, top score review:

| Score | Any M+S: 10% / 20% | Top severity quartile: 10% / 20% |
|---|---|---|
| FA-v2 | 1,3043 / 1,3406 | **0,5829 / 0,5371** |
| Presence | 0,8696 / 1,0870 | 0,5829 / 0,8286 |
| Partition-v1 | 1,3768 / 1,4130 | **1,4914 / 1,4829** |
| Partition-v2 | 1,3768 / 1,3768 | 0,5829 / 0,7829 |

Random enrichment reference là 1. Binary enrichment có lợi không chứng minh FA-v2 ưu tiên nhóm severity cao nhất; hai top-severity point estimates của FA-v2 đều dưới 1. Stage 2 cũng không cho pooled severe-error enrichment rõ vượt 1 ở FA-v2 (0,8429/0,9833). Đây là kết quả mô tả của các rankings đã định, không thêm kiểm định significance. [Evidence: E4_enrichment](../outputs/stage3_h1_v2/reporting/e1_e5_evidence.json).

### Internal F1 có blindspots về volume và cấu trúc

F1≥1−1e−12 nhưng missing-volume>0,10 xuất hiện ở **9/320 validation model-bone records** và **7/800 held-out records**. Mỗi split có 4 cases và 4 bone units phân biệt; không cộng chúng thành 16 bệnh nhân.

Ví dụ held-out `302/femur/member_03`: F1=1 với **8 GT instances, 6 predictions, 2 merges, 8 splits**, M+S risk=1,25 và missing-volume=43,56%. GT best-IoU assignment của evaluator nội bộ không bắt buộc one-to-one; F1 này không thể thay cho structural risk hoặc volume coverage. Kết luận chỉ áp dụng evaluator đã dùng, không suy rộng sang official PENGWIN evaluator. [Evidence: E4_blindspots và E4_blindspot_counts](../outputs/stage3_h1_v2/reporting/e1_e5_evidence.json).

<a id="e5"></a>

## E5: lợi ích ranking toàn đường cong không đồng nghĩa fixed-budget utility

Pooled held-out member-mean FA-v2/presence retained M+S risk ở coverage 0,5 là **0,3780/0,5774**, nhưng ở coverage 0,8 là **0,6041/0,6044**. AURC gộp toàn coverage nên có thể tốt hơn trong khi budget review 20% gần như không cải thiện.

Supplementary paired differences `FA-v2−presence`, pooled held-out member-mean:

| Risk | Review 10%: Δ [95% CI] | Review 20%: Δ [95% CI] |
|---|---|---|
| Merge+split | −0,02653 [−0,05710; 0,01200] | −0,00027 [−0,06011; 0,05128] |
| 1−F1 | −0,00769 [−0,02047; 0,00485] | +0,00608 [−0,01401; 0,01952] |
| Missing volume | **+0,00645 [0,00173; 0,01365]** | **+0,00593 [0,00240; 0,01242]** |

Các CI M+S và 1−F1 cắt 0; missing-volume nghiêng về presence ở cả budgets. Đây là marginal exploratory intervals, không hiệu chỉnh familywise và không thay primary E3. Stage 2 review 20% có M+S retained risk **0,6146** với FA-v2 so với **0,5799** với presence, nên không gọi review utility là robust qua risk sources.

Review 40/200 bones tương ứng **34 CTs với FA-v2, 32 presence, 37 partition-v1, 33 partition-v2**. Đây là expected distinct CT count dưới cutoff ties, không phải measured reading time hoặc equal-CT-cost comparison. Không mặc định review sẽ sửa lỗi hoàn hảo.

Giữ nguyên các ví dụ E5 theo rule: `176/right_hip`, `160/sacrum`, `187/left_hip`. Ca cuối có cả ba member F1=0, missing-volume=1 nhưng disagreement=0 và M+S risk=0: lỗi bỏ sót toàn anatomy không được structural endpoint này phản ánh. Không thay bằng ví dụ thuận lợi hơn. [Hình E5](../outputs/stage3_h1_v2/e5_supplement_v1/figures/qualitative.pdf).

Runtime E3 ghi median inference **bao gồm model load** khoảng 80,03–83,18 giây/case/model; scoring+evaluation CPU median 19,68 giây/case. Không có historical model-load/cleanup/baseline timings riêng. Không gọi 19,68 giây là overhead riêng của VI hoặc speedup so với score khác. [Evidence: E5_paired_budget, E5_fixed_budget, E5_runtime](../outputs/stage3_h1_v2/reporting/e1_e5_evidence.json).

## Claims và giới hạn

| ID | Claim có thể dùng | Evidence | Trạng thái / giới hạn |
|---|---|---|---|
| C1a | Foreground và partition contributions có thể tách định lượng | E1 decomposition; E2 controlled responses | Supported trong setup này; current per-bone khác historical whole-case |
| C1b | VI/log N giảm dưới count replication, còn VI/H bất biến | E2c 240 tables, r=1…16 | Supported cho count algebra; không phải image-resolution invariance |
| C2 | FA-v2 đạt criterion pooled held-out so với presence/entropy | E3 hai primary contrasts; Stage 2 sensitivity | Supported cho 100 cases/backend này; formal registration chưa được xác minh |
| C2-limit | FA-v2 hoặc joint normalization luôn tốt nhất | Partition-only/Hungarian baselines; E4 contrasts | **Không supported**; không đổi primary post hoc |
| C3a | Annotation attribution và normalization cần được kiểm soát riêng | E4 case-level 2×2 và outside-scope ratios | Supported; chưa chứng minh scope restriction tự làm ranking tốt hơn |
| C3b | Binary prevalence/enrichment không thay cho severity prioritization | E4 prevalence và quartile enrichment | Supported; FA-v2 severe-error enrichment pooled dưới random ở member-mean |
| C3c | Internal F1 có thể che volume/structural errors | Corrected blindspot CSV, fragment coverage | Supported cho evaluator nội bộ; không gọi là lỗi của evaluator official |
| C3d | AURC tốt không bảo đảm review utility ở budget cụ thể | E5 original + supplementary CIs, CT costs | Supported; chưa chứng minh annotation savings hoặc clinical benefit |
| C4-support | Artifact chain hỗ trợ reproducibility của nghiên cứu backend này | Manifests, code/config hashes, correction receipt | Supported trong phạm vi files đã kiểm tra; không tái đánh giá backend fixes/gateway lịch sử hoặc độc lập tái lập pixel-level |
| External | Chuyển giao sang dataset/backend khác | E6 chưa có kết quả được audit ở đây | **Chưa có bằng chứng** |


## Partition v3 và v4

Đây là development sau E3 trên cùng 40 CT validation, không phải xác nhận độc lập. Primary E3 và các baselines vẫn giữ nguyên.

| Nhánh | Estimand | Candidate / P1 AURC | Δ candidate−P1 [95% CI] |
|---|---|---|---|
| v3: ridge(P1,G,presence) | Pooled OOF bone AURC | 0,41821 / 0,27059 | +0,14761 [0,05273; 0,24788] |
| v4: pairwise(P1,G), bỏ presence | Mean outer-fold bone AURC | 0,28873 / 0,28938 | −0,00064 [−0,01826; 0,00868] |

V3 raw G có AURC **0,26895**, so P1 Δ **−0,00164 [−0,01485; 0,01076]**. V4 đổi objective và cách tổng hợp folds để kiểm tra một ablation cụ thể, nhưng CI vẫn cắt 0. Không so AURC tuyệt đối v3/v4 như cùng estimand. Các CI conditional trên fixed folds/models/OOF scores, không bao trùm training uncertainty.

Nguồn: [v3 comparison](../outputs/partition_v3_development_v1/comparison.csv), [v3 contrasts](../outputs/partition_v3_development_v1/paired_differences.csv), [v4 comparison](../outputs/partition_v4_development_v1/comparison.csv), [v4 contrasts](../outputs/partition_v4_development_v1/paired_differences.csv). **Chưa giải quyết được điểm yếu so với partition-only; không làm baseline P1 tệ đi để tạo ưu thế.**

<a id="e6"></a>

## E6: RibFrac và revision tăng tốc

Đã có inventory **420 development + 80 external pairs**, preprocessing/cache và `handoff.json` tại `/mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6-fast-v1`. Handoff ghi chuyển member01/02 từ epoch 30, member03 bắt đầu từ đầu; giữ nguyên checkpoints nguồn. Đây là chuyển giao fixed scores sang rib-fracture lesion instances với backend mới, không chuyển PENGWIN checkpoints.

Snapshot đọc log lúc khoảng **19:31 ngày 22/11/2026, giờ Việt Nam**: member01 bắt đầu epoch 81, member02 bắt đầu epoch 75; epoch vừa hoàn tất mất khoảng 219,71 / 260,77 giây. Hai training manifests ghi `running`. Đây là snapshot từ log/manifests; không thay kiểm tra scheduler hiện thời và không phải ETA cam kết. Chưa có external result để kết luận transfer.

Batch revision giữ gần 500 train patches/epoch: batch6 dùng 83 train/17 validation iterations (498/102 patches), batch5 dùng 100/20 (500/100). Giữ 1.000 epochs, LR schedule theo epoch và final-checkpoint policy. Cùng patch exposure không đồng nghĩa cùng optimizer trajectory; members warmstart có prefix batch2 phải được ghi trong Methods. Mỗi GPU task dùng một GPU Slurm cấp, tối đa hai tasks đồng thời.

## Manuscript và đánh giá khả năng nộp

Bằng chứng hiện tại hỗ trợ một nghiên cứu phân rã và đánh giá failure detection, chưa hỗ trợ tuyên bố phương pháp vượt mọi baseline, giảm chi phí review lâm sàng hoặc transfer thành công. Rà soát EMBC và bản nháp Results/Discussion được giữ trong archive làm nguồn cho bước viết tiếp; nhận định đó không phải acceptance guarantee. Quy định/deadline venue cần kiểm tra lại trước submission.

## Kiểm tra tính toàn vẹn và nơi lưu bằng chứng

- Sau khi gộp docs: **211 unit tests PASS** trong project environment (22,779 giây); công cụ dựng E2 matched-cohort figure chạy thành công tại output path mới.
- E1–E5 auditor xác minh **111 output hashes + 142 cache hashes** và evidence sidecar. Audit không thay thế việc đọc lại toàn bộ CT/checkpoints hoặc tái lập pixel-level.
- Run-bound config hashes được kiểm tra; chưa xác minh formal preregistration hoặc timestamped commit trước held-out. Không dùng từ preregistered chỉ từ các hashes.
- [Evidence sidecar](../outputs/stage3_h1_v2/reporting/e1_e5_evidence.json) và [E2 matched-cohort figure](../outputs/stage3_h1_v2/reporting/e2_matched_merge.pdf) nằm trong `outputs/`, không tạo lại thư mục reports trong docs.
- Dataset: `dataset -> ../dataset`; bulk E6 roots nằm ngoài repo. [Manifest copy v1](../outputs/local_copy_v1/manifest.json) giữ nguồn/đích và SHA-256. Bốn checkpoint smoke backend cũ giữ symlink tới bản lưu cleanup trước; không xóa đích khi còn cần symlink.

Toàn bộ **51 file tài liệu trước khi gộp**, gồm reports, literature, rules, storage map và hình gốc, được giữ nguyên byte tại [archive](../outputs/archive/documentation_20261122T123244Z/__docs__/). Archive dùng để tra cứu bằng chứng; không dùng các status/command cũ làm hướng dẫn hiện hành.

## Rà soát toàn dự án trước khi cân nhắc chạy lại — 22/11/2026

### Material Passport

- Origin Skill / Mode: `academic-research-suite` → experiment-agent, plan; đối chiếu artifacts và code bằng kiểm tra chỉ đọc.
- Verification: số liệu dưới đây đã kiểm tra từ CSV/code/log hiện có hoặc phantom nhỏ; các cải tiến đề xuất **chưa được xác nhận**. Không chạy lại training, inference, dataset-wide processing hay bootstrap.
- Version: `project_review_20261122_v1`. Giữ ba file docs; protocol E1–E6/v3/v4 không đổi.

### Phương pháp: thêm thông tin có ích trước khi đổi trọng số

P1 và G có Spearman **0,94452** trên 80 validation bones. Cả hai cùng bằng 0 ở 27/80 units; trong nhóm này missing-volume vẫn có thể tới **0,43315**. V3 chọn ridge theo MSE/log-risk, khác ranking objective; v4 đã xử lý phần objective/aggregation nhưng ΔAURC so P1 vẫn **−0,000643 [−0,018265; 0,008681]**. Chưa thấy lỗi công thức graph/CV đủ để giải thích thất bại. Dữ liệu ủng hộ cách giải thích rằng G bổ sung ít thông tin độc lập.

Một baseline còn thiếu trong các heads là **absolute predicted instance count** (`predicted_count_mean`), khác `count_variance` đang có. Phân tích hậu nghiệm chỉ dùng `outputs/partition_v3_development_v1/oof_scores.csv`, gọi `aurc_batch`, không fit/bootstrap:

| Validation, member-mean M+S | P1 | G | Predicted count mean |
|---|---:|---:|---:|
| Pooled, 80 bones | 0,27059 | 0,26895 | 0,26328 |
| Sacrum, 20 | 0,05749 | 0,05989 | 0,05892 |
| Left hip, 20 | 0,18506 | 0,18654 | 0,18056 |
| Right hip, 20 | 0,45009 | 0,40918 | 0,40613 |
| Femur, 20 | 0,65197 | 0,65369 | 0,71357 |

Đây là point estimates **không có CI**, không phải ưu thế đã xác nhận. Count có thể chỉ phản ánh độ phức tạp/anatomy; femur kém hơn rõ về point estimate. Có cơ sở cho đúng một ablation nhỏ **P1+count**, giữ nguyên P1/G/count-only/count-variance/v4 làm controls và báo strata. Không gọi count là uncertainty metric mới.

Nếu phát triển graph tiếp, significant overlap edges và unmatched-object fractions khác với effective-degree hiện tại. Tuy nhiên unmatched events rất thưa (first-direction 6/240 pairs, reverse 7/240), nên chưa có cơ sở cho head nhiều tham số. Hướng này ưu tiên sau ablation count, không mở một cuộc tìm feature tự do. [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Siddiqui_Uncertainty_Estimation_in_Instance_Segmentation_With_Star-Convex_Shapes_WACV_2024_paper.pdf) cung cấp cách tách detection/shape certainty theo object correspondence; chuyển sang fracture CT vẫn là giả thuyết của dự án.

### Endpoint: không đồng nhất số split với mức độ lỗi

`instances.py` và `e6_evaluation.py` dùng many-to-one best-IoU matching, threshold 0,1, và split tính mọi positive-overlap prediction ID. Tiny CPU phantoms trên evaluator E6, chưa áp component pruning PENGWIN:

| Tình huống | F1 | M+S risk | Dice |
|---|---:|---:|---:|
| Hai GT bằng nhau bị gộp thành một prediction | 1 | 0,5 | 1 |
| Một GT 64 voxels tách thành 63+1 | 0,6667 | 1 | 1 |
| Cùng GT tách thành 32+32 | 0,6667 | 1 | 1 |
| GT có foreground, prediction rỗng | 0 | 0 | 0 |

Đây là tính chất của endpoint đã công bố, không phải bằng chứng toàn bộ evaluator sai. Cần supplementary sensitivity với one-to-one matching, significant overlap và missing-volume; giữ nguyên primary cũ. Threshold/định nghĩa severity phải chọn có lý do trước khi xem kết quả mới. [Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z) khuyến nghị chọn localization, assignment và detection/segmentation metrics theo domain interest; bài không xác nhận một threshold mới cụ thể cho project.

### Omission và review policy

E3 `187/left_hip` có n_GT=5, cả ba members **và Stage 2** cùng rỗng: 12 scores bằng 0, M+S=0, missing-volume=1. Một foreground-disagreement normalization khác hoặc thêm đúng Stage 2 vào ensemble không giải quyết được ca này. Expected-anatomy flag/image evidence/reference evidence là một kênh khác; [RCA](https://arxiv.org/abs/1702.03407) là ví dụ dùng reference có nhãn để ước tính chất lượng mà không cần GT của ca mới, không phải giải pháp đã chứng minh cho fracture.

**Equal-CT policy và omission guard đã có ở v3**, không cần tái triển khai như tính năng mới. Nhưng validation có 0 flagged CTs và 0 complete-omission-positive CTs theo member-mean endpoint nên chưa kiểm chứng sensitivity của guard. Ridge−P1 tại budget 20% CT có Δ retained risk **−0,003727 [−0,034040; 0,037393]**: chưa có lợi ích chắc chắn. V4 không đánh giá CT policy. Bước hữu ích là kiểm chứng policy P1+guard đã đóng băng trên cohort độc lập, với cùng ngân sách CT và endpoint omission riêng. Không chuyển expected-positive-anatomy rule sang RibFrac có ca âm tính thật.

### E6: chất lượng training và khả năng tăng tốc

Không thấy lỗi training chí mạng trong phạm vi rà soát; 9 source hashes/member của hai manifests hiện có khớp. Cache/budget revision đã rút ngắn epochs theo logs; last-ten-epoch medians tại snapshot rà soát khoảng **223,7 / 256,8 giây**, không phải ETA hoặc speedup riêng do batch.

Điểm cần thận trọng khi chạy lại: batch2→6 giữ gần số patches nhưng đổi **250→83 optimizer updates/epoch**. Với nnU-Net oversampling nominal 0,33 và deterministic rounding thực tế, batch2 có **1/2** samples được ép foreground, batch6 có **2/6** (batch5: **2/5**). Đây là tỷ lệ guaranteed-foreground slots, không phải tỷ lệ patches thực tế chứa foreground. Batch Dice, momentum/weight-decay per update cũng thay đổi. Vì vậy không mặc định cùng patch exposure sẽ giữ nguyên chất lượng.

Nếu cần revision training mới, làm một-seed internal-validation pilot từ cùng warmstart ở root riêng, giữ NPY cho mọi arm: batch2 baseline, batch6 hiện tại và batch6 có guaranteed-foreground exposure khớp baseline. Đo full-volume lesion outcomes trên tập internal-validation cố định, không chỉ patch pseudo-Dice, trước khi trả chi phí ba seeds/1.000 epochs. Không đổi current mixed-batch campaign giữa chừng.

Backend hiện collapse instance IDs thành binary target rồi threshold0,5 + 26-connectivity; không được supervision để phân biệt hai lesion chạm nhau. Trước khi thay kiến trúc, kiểm tra tần suất GT lesions chạm nhau và lỗi internal validation; watershed/boundary/distance chỉ là backend ablation riêng nếu có bằng chứng cần thiết.

CSV inventory xác nhận **336 train / 84 internal-val đều không có negative CT**, còn external có **20/80 negatives**, median lesion count 9/9/5. Structural endpoints chỉ xác định trên **60 external positives**; F1/Dice/FP cần báo cả 80. Thêm bảng FP instances/negative CT và burden theo positive/negative; không lấy external negatives để tune. [Official RibFrac protocol](https://ribfrac.grand-challenge.org/evaluation/) dùng detection FROC và IoU>0,2, khác evaluator nội bộ IoU≥0,1; không gắn nhãn official performance cho bảng nội bộ.

Về runtime, single-process augmentation và compile-off là lựa chọn phục hồi RNG hiện hành. Profile data/augmentation wait, H2D, train, validation, checkpoint trước khi bật workers/compile. Có thể đưa full cache-hash audit sang CPU predecessor job thay vì giữ GPU trong lúc đọc cache, nhưng cần ràng buộc verified receipt và bảo đảm không có writer sửa cache giữa audit và train; không bỏ hash check để tăng tốc. Heartbeat theo epoch có thể giúp monitoring; mọi thay đổi trainer/prefetch phải ở revision riêng.

### Confirmation và tái lập

- Có **110 candidate CTs** còn lại theo phép trừ split; `confirmation_candidates.json` vẫn ghi `exposure_audit_completed=false`, `confirmation_ready=false`. Membership không chứng minh chưa xem model-error outcomes. Cần exposure audit trước khi dùng làm fresh confirmation. Local 50-case `test` là phần hold-back từ public labeled set, không tự đồng nhất với official challenge hidden test và vẫn ngoài scope hiện hành.
- Source/config/tests hiện đều được Git track ở snapshot commit `f7c8c62`; không còn dùng ghi chú cũ rằng tất cả code chưa track. Nhưng ABBC gitlink thiếu `.gitmodules` và checkout không có remote `origin`; `git submodule status` thất bại. Cần khôi phục mapping/URL upstream đã xác minh và kiểm tra clean-clone bootstrap, giữ đúng pin+patch.
- Evidence auditor ban đầu FAIL vì duy nhất cờ `e3_config_tracked_in_current_git_index` đổi false→true sau Git update. Đã lưu bản trước rồi tái tạo reporting sidecar; **111 output hashes + 142 cache hashes vẫn PASS**, số liệu khoa học không đổi. Cờ Git hiện tại không chứng minh chronology/preregistration trước held-out.
- Không sửa training/scoring/evaluator code, configs hoặc jobs trong lần review. Thay đổi chỉ ở bản tổng hợp, kế hoạch đề xuất và cờ metadata reporting vừa mô tả. Các hướng cần code/full compute đều chưa triển khai.

## Implementation follow-up sau rà soát — 22/11/2026

**Status: implementation/smoke complete; full follow-up chưa chạy.** Sau khi tác giả đồng ý, đã thêm các nhánh riêng dưới đây. Original E1–E6/v3/v4 code/configs và các kết quả cũ giữ nguyên. Không submit full jobs trong phiên implementation.

| Thành phần | Đã triển khai | Kiểm tra |
|---|---|---|
| P1+count | Hai prediction-only features, v4 grouped 5×3 CV/objective/grid, frozen controls, 10.000 paired CT bootstraps, anatomy và diagnostic GT-complexity tables | Real preflight 40 CT/80 bones; synthetic 20 CT/40 bones, 32 draws COMPLETE |
| Evaluator sensitivity | One-to-one matching, cả hai significant-overlap thresholds, missing-volume/original endpoints, equal-CT P1/guard/FA-v2/presence và omission diagnostics | Real metadata preflight 40 validation +100 held-out; synthetic 10 CT/20 bones, 32 draws COMPLETE; không đọc real volumes |
| Reproducibility preflight | Khôi phục `.gitmodules`, verify ABBC pin+exact patch, scan bounded metadata cho 110 candidate IDs | 132 metadata files, 0 recorded candidate outcomes; unknown exposure và `confirmation_ready=false` giữ nguyên |
| E6 cohort report | Hash-verified completed CSVs, positive/negative FP burden, F1/Dice và structural eligibility; output riêng | Synthetic complete-source/round-trip/idempotency/corruption tests PASS; chờ E6 thật complete |
| Master Slurm launcher | CPU preflight → hai synthetic smokes → hai full jobs với `afterok` | Fake-scheduler failure/dependency tests, shell syntax và Slurm `--test-only` PASS; không submit full |

**Validation:** toàn bộ **263 unit tests PASS (29,132 giây)** trong `exchange-stage2`; E1–E5 evidence audit tiếp tục **PASS 111 output hashes +142 cache hashes**. Hai default synthetic smoke receipts được kiểm tra độc lập **22 output/cache hashes** cùng code hashes. Hai active E6 training manifests có **9/9 source hashes mỗi member** vẫn khớp. Không đổi số optimizer updates, sampling, threshold hoặc checkpoint policy của E6.

Count smoke ban đầu gặp scipy line-search `ABNORMAL` ở analytic gradient khoảng `6e−9`. Code mới tự xác minh stationarity và ghi native solver status, chỉ chấp nhận dưới `1e−7`; regression tests từ chối nghiệm không stationary hoặc Jacobian giả. Failed smoke được giữ ở `/tmp/exchange-count-smoke-v1`; successful final smoke ở `outputs/smoke_count_ablation/`. Việc này không sửa optimizer/control v4 đã frozen.

Trong review chéo, cache/resume evaluator được bổ sung payload digest và incremental manifest hashes; output overrides chỉ dùng namespace riêng; omission CI có explicit status khi bootstrap thiếu positive CT hoặc guard infeasible. Không bỏ các draws đó để tạo CI thuận lợi.

**Agent-typed checks:** project-environment `unittest discover -s tests -q`, metadata `check` của hai runners, reproducibility preflight, synthetic `smoke` của hai runners, shell `bash -n`, `sbatch --test-only` và audit E1–E5. Logs tests: `/tmp/exchange-followup-tests.log`. Slurm test-only là kiểm tra admission, không phải jobs đã chạy hay ETA.

**User-typed full command:** [runbook và resources](exps.md#follow-up-sau-rà-soát-slurm-cpu). Master: `bash scripts/submit_research_followup.sh full`. Optional E6 cohort report chỉ sau successful evaluation, ví dụ dependency job2925 của chain hiện tại. Không chạy đồng thời master và launchers từng nhánh vào cùng roots.

**Next step:** tác giả chạy master command; kiểm tra complete manifests và paired differences, giữ cả negative/undefined findings. Count ablation là development và evaluator dùng lại E3 đã xem; các jobs này không xác nhận độc lập hay chứng minh cải thiện trước khi có kết quả. Training pilot/instance-backend revision vẫn là bước có điều kiện sau diagnostics, chưa được thêm vào current E6 campaign.


## Kiểm tra full follow-up — 23/11/2026

Slurm preflight2937, smokes2938/2939 và full2940/2941 đều COMPLETED, exit0:0. Count full34 giây; evaluator full44 phút39 giây. Manifests full complete; kiểm tra độc lập15 count output hashes +148 evaluator output/cache hashes và producer source hashes PASS.

Count ablation primary mean-fold AURC: P1=0,28938; count-only=0,27860; P1+count=0,28114. Candidate−P1 **−0,00824 [−0,03813;0,01151]**: point tốt hơn nhưng CI cắt0, chưa xác nhận cải thiện; không thắng count-only về point estimate. Đây vẫn là validation development.

Held-out supplementary FA-v2−P1: original M+S **+0,05914 [0,03931;0,08012]**; one-to-one1−F1 **−0,00164 [−0,02684;0,02799]**; significant-overlap tau0,01 **+0,00326 [−0,00792;0,01428]**, tau0,1 **−0,00087 [−0,00548;0,00349]**. Lợi thế P1 trên original endpoint không còn rõ ở supplementary endpoints; điều này không chứng minh FA-v2 thắng hoặc cho phép thay primary. Không so absolute AURC giữa risk definitions để tuyên bố cải thiện model.

Guard+P1 soP1, equalCT20% budget, missing-volume delta **−0,00429 [−0,01213;0,00810]**, chưa có lợi ích chắc chắn. Heldout omission endpoint có2 positiveCTs; guard sensitivity0,5/1,0 tại10%/20% review nhưng CIs undefined vì bootstrap códraw không positives; mẫu quá nhỏ để khẳng định utility.

E6 snapshot log10:45 ngày23/11: member01 bắt đầu epoch320/1000, member02 epoch298/1000; Slurm2923_0/1 RUNNING, member03 chờ JobArrayTaskLimit; infer2924/eval2925 chờ dependency. Chưa có external result hoặc cohort report. Không submit/restart jobs trong lần check.

Next step: giữ count result là mixed/negative, dừng tuning fusion không giới hạn; dùng sensitivity làm bằng chứng endpoint-dependence; hoàn tất E6 và exposure audit trước bất kỳ fresh confirmation nào.

## Pipeline EMBC: literature, policy và manuscript — 23/11/2026

Đã đọc6 full scientific manuscripts có publication EMBC được xác minh (3năm2023,2năm2024,1năm2025), cùng một study âm tính2022 để tham khảo. Một2024paper dùng full preprint10trang của tác giả vì chưa truy cập final5trang; giới hạn phiên bản được ghi rõ. Review không ước lượng acceptance probability hoặc suy từ accepted-only rằng yêu cầu khoa học có thể bỏ qua. Matrix và source ledgers ở `paper/review/embc_acceptance_synthesis.md`, `embc_recent_fulltext.*`, `embc_comparison_fulltext.*`.

Đã triển khai dual-channel CT triage prediction-only, calibration fold-local, candidate và baseline cùng omission guard; chi tiết và command Slurm trong `exps.md`. **8 tests mới PASS; toàn suite271 tests PASS (28,671giây)**. Real metadata preflight40validation/100heldout pass, synthetic smoke và Slurm `--test-only` pass; E1–E5 audit111/142 tiếp tụcPASS. Chưa chạy full policy mới hoặc thay E6.

Manuscript được chuẩn bị bằng template IEEE trong `paper/`, tách completed evidence khỏi pending policy/E6; các integrity/review receipts và trạng thái pipeline lưu dưới cùng thư mục. Bản review ẩn danh dùng khi chưa có author metadata, không tự bịa authors/funding/ethics approval.
