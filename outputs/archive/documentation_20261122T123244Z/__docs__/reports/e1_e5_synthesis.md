# Tổng hợp khoa học E1–E5

**Status: complete cho tổng hợp và audit báo cáo, 20/11/2026.** E1, E2, E3, E4–E5 và E5 supplementary đều có manifest full `complete`. Báo cáo này chỉ đọc artifacts hiện có; không chạy lại inference, scoring hay bootstrap. Đây là bản tổng hợp bằng chứng, chưa phải quyết định chấp nhận paper hoặc sign-off các gateway lịch sử.

**Kết luận chính:** FA-v2 đạt tiêu chí pooled held-out so với presence và mean entropy. Kết quả không chứng minh FA-v2 là score tốt nhất, normalization mới luôn tốt hơn, hoặc review 10–20% xương đem lại lợi ích thực tế chắc chắn. E4/E5 cho thấy khác biệt giữa ranking trên toàn coverage, ưu tiên lỗi nặng, lỗi bỏ sót và chi phí review.

## 1. Bộ bằng chứng và phạm vi kiểm tra

| Thực nghiệm | Dữ liệu / đơn vị | Vai trò |
|---|---|---|
| E1 | 40 validation cases: 20 pelvic + 20 femur; 80 bones = 60 + 20 | Chẩn đoán và thiết kế v2; exploratory |
| E2 | 20 validation GT cases; 40 bones; 527 responses, 452 eligible | Controlled mechanism check, không phải model failure detection |
| E3 | 100 held-out cases: 50 + 50; 200 bones = 150 + 50; bốn models, 400 inference records | Đánh giá tiêu chí pooled đã mã hóa trong config |
| E4 | Hai split riêng; 140 cases, 280 bones, 4.000 fragment/model rows | Sensitivity theo thiết kế E4; CI secondary |
| E5 gốc | Hai bone review budgets và ba ví dụ đã chọn bằng quy tắc | Utility và failure modes |
| E5 supplementary | Cùng dữ liệu, thêm fixed-budget CIs và omission endpoints | **Exploratory post-heldout**, không tạo primary mới |

Audit hiện tại xác minh **111 recorded output hashes và 142 recorded cache hashes**, chữ ký execution, code hashes được ghi nhận, chuỗi parent manifests và reporting correction. E1/E2 caches không có hash riêng trong các manifest đó nên không được cộng vào 142. Kiểm tra này không phải đọc lại mọi raw volume hoặc checkpoint. Báo cáo E4 đã sửa tên cột nhưng giữ execution signature và code gốc; audit dùng correction receipt để kiểm tra code exporter sau sửa.

Evidence sidecar: [e1_e5_evidence.json](e1_e5_evidence.json). Các nhóm `extracted` ghi source path, source SHA-256, filters và exact CSV rows; các thống kê tính lại từ CSV nằm ở `derived`. Có thể tái kiểm tra bằng [audit_e1_e5_evidence.py](../../scripts/audit_e1_e5_evidence.py).

**Giới hạn chronology:** run-bound hash của config E3 được xác minh. Config hiện không nằm trong Git index; audit chưa xác minh một đăng ký độc lập hoặc timestamped commit trước held-out. Dùng “criterion encoded in the held-out configuration” hoặc diễn tả chronology có nguồn, không khẳng định formal preregistration chỉ từ hash. E5 supplementary được đánh dấu rõ là thêm sau khi xem held-out. Venue/deadline và quyết định G2/H1 lịch sử không được xác minh hoặc tự thay đổi trong lần tổng hợp này.

## 2. E1: tín hiệu partition có thật, nhưng phải phân biệt với whole-case lịch sử

E1 per-bone dùng cùng cleanup và units với E3: tính score không dùng GT voxels, rồi đánh giá risk từng model trước khi mean ba member. Presence là foreground Jaccard disagreement. Weighted partition terms là `w·VI/log(N)` và `w·VI/H`, với `N` là số voxel shared foreground, `H` là joint entropy, `w` là shared/union foreground fraction. Mean ba cặp member được lấy trước khi tóm tắt qua bone units.

| Quantity, pooled 80 bones | Median | Maximum |
|---|---:|---:|
| Raw VI, nats | 0,014834 | 0,601793 |
| Weighted partition v1 | 0,001164 | 0,047092 |
| Weighted partition v2 | 0,032116 | 0,988552 |
| Partition share trong FA-v1 | 6,9581% | 76,8585% |
| Partition share trong FA-v2 | 58,8711% | 98,8583% |

Raw mean-pair VI ≤1e−12 ở **27/80 bones (33,75%)**. Vì nhiều bones khác vẫn có VI dương, việc cả ensemble hoàn toàn đồng ý về partition không giải thích toàn bộ dữ liệu. Chuẩn hóa joint entropy làm contribution tăng; việc tăng magnitude không tự chứng minh ranking tốt nhất.

Các số **“share median khoảng 2%, max contribution <0,02, rank correlation 0,997” thuộc whole-case analysis lịch sử**. Không thay chúng vào current per-bone E1: đơn vị, scope và AURC convention đã khác. Các source chính cho E1 hiện tại là `tables/decomposition.csv` và `validation/*`, không phải bảng H1 lịch sử. [Evidence: E1_decomposition, E1_validation_contrasts](e1_e5_evidence.json).

## 3. E2: thành phần phản ứng đúng trong perturbations, có giới hạn severity

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

[Hình matched-cohort mới cho báo cáo](figures/e2_matched_merge.pdf) giữ cùng bones qua mọi severity trong mỗi panel, hiển thị trajectories và median, không có CI hay claim clinical severity. Hình được dựng từ CSV đã hash-verify; các figures gốc của E2 không bị thay thế.

E2c tái dựng **240 E1 contingencies**: 154 positive-VI + 86 zero-VI pairs. Nhân mọi union counts, kể cả foreground mismatch, với `r={1,2,4,8,16}` giữ VI, H, presence, weight và Q2 bất biến; Q1 đi theo `VI/log(rN)`. Residuals ghi trong summary bằng 0. Có 775 positive-VI/phantom replication rows và 430 zero-VI rows. Đây là phép thử đếm chính xác, không phải CT resampling hoặc chứng minh invariance theo độ phân giải ảnh. [Evidence: E2 summaries và derived/matched_merge_q_joint](e1_e5_evidence.json).

## 4. E3: tiêu chí chính đạt, không phải thắng mọi baseline

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

Partition-only tốt hơn FA-v2 trong pooled held-out M+S: FA-v2−partition-v1 **+0,05914 [0,03921; 0,08032]**, FA-v2−partition-v2 **+0,03333 [0,01535; 0,05265]**. Đây là secondary contrasts, không chuyển primary sang hàng thắng sau khi xem held-out. Hungarian có point AURC 1−F1 thấp nhất trong bảng, nên không gọi FA-v2 là detector tốt nhất cho mọi loại failure. [Evidence: E3_main_table và E3_structural_contrasts](e1_e5_evidence.json).

## 5. E4: bốn điều kiện đánh giá phải được tách riêng

### 5.1 Normalization × annotation scope

E4a là **case-level 2×2**. Trong mỗi model, risk bằng tổng merge+split trên anatomy được annotate chia tổng GT instance count, rồi mean ba members. Không so absolute AURC E4 với E3 per-bone để quy phần chênh cho scope; cả unit weighting và risk aggregation khác nhau.

Pooled held-out case-level contrasts:

| Controlled change | Member-mean ΔAURC [95% CI] | Stage 2 ΔAURC [95% CI] |
|---|---|---|
| Joint H thay log N, all scope | −0,03036 [−0,07766; 0,01558] | −0,03756 [−0,08699; 0,01119] |
| Joint H thay log N, annotated scope | **−0,07030 [−0,13079; −0,01190]** | **−0,06979 [−0,13052; −0,01147]** |
| Annotated thay all scope, log N | +0,05529 [−0,02067; 0,13445] | +0,02081 [−0,06900; 0,11039] |
| Annotated thay all scope, joint H | +0,01535 [−0,05856; 0,09291] | −0,01141 [−0,09585; 0,06838] |

Joint normalization có lợi trong pooled annotated-scope case ranking; **annotation restriction tự nó chưa cho thấy AURC improvement**. Các CIs của annotated normalization ở cả hai scan-type strata, cho cả hai risk sources, cũng cắt 0. Các CI trên là marginal secondary intervals, không phải bốn primary tests mới.

### 5.2 Disagreement ngoài annotation scope

Quantity sau là mean qua cases của mean ba pair ratios `|D∩O|/|D|`, với `D` là foreground symmetric difference và `O` là vùng prediction gán anatomy ngoài scope:

| Split | Pelvic mean | Femur mean |
|---|---:|---:|
| Validation | 31,62% | 87,07% |
| Held-out | 29,66% | 86,55% |

Held-out femur median của case means là 93,07%. Đây là **predicted anatomy attribution**, không phải GT xác nhận mọi voxel đó là bone chưa annotate. Mean pooled case ratios là 58,11%, trong khi ratio của summed voxel counts là 79,69%; hai cách aggregation trả lời hai câu hỏi khác nhau. Disagreement ngoài GT foreground được lưu riêng vì còn chứa boundary errors và false positives. [Evidence: derived/annotation_scope_case_summary; scope pair/outside-GT CSVs](e1_e5_evidence.json).

### 5.3 Binary prevalence khác ưu tiên lỗi nặng

Any-M+S prevalence held-out là **138/200=69%** ở member-mean và **125/200=62,5%** ở Stage 2, cho binary enrichment ceiling 1,4493 và 1,6000. Femur member-mean có **48/50=96%**, ceiling chỉ 1,0417. Không dùng “92,5% ca” của historical whole-case analysis để mô tả các denominator per-bone này.

Enrichment trên pooled held-out member-mean, top score review:

| Score | Any M+S: 10% / 20% | Top severity quartile: 10% / 20% |
|---|---|---|
| FA-v2 | 1,3043 / 1,3406 | **0,5829 / 0,5371** |
| Presence | 0,8696 / 1,0870 | 0,5829 / 0,8286 |
| Partition-v1 | 1,3768 / 1,4130 | **1,4914 / 1,4829** |
| Partition-v2 | 1,3768 / 1,3768 | 0,5829 / 0,7829 |

Random enrichment reference là 1. Binary enrichment có lợi không chứng minh FA-v2 ưu tiên nhóm severity cao nhất; hai top-severity point estimates của FA-v2 đều dưới 1. Stage 2 cũng không cho pooled severe-error enrichment rõ vượt 1 ở FA-v2 (0,8429/0,9833). Đây là kết quả mô tả của các rankings đã định, không thêm kiểm định significance. [Evidence: E4_enrichment](e1_e5_evidence.json).

### 5.4 Internal F1 có blindspots về volume và cấu trúc

F1≥1−1e−12 nhưng missing-volume>0,10 xuất hiện ở **9/320 validation model-bone records** và **7/800 held-out records**. Mỗi split có 4 cases và 4 bone units phân biệt; không cộng chúng thành 16 bệnh nhân.

Ví dụ held-out `302/femur/member_03`: F1=1 với **8 GT instances, 6 predictions, 2 merges, 8 splits**, M+S risk=1,25 và missing-volume=43,56%. GT best-IoU assignment của evaluator nội bộ không bắt buộc one-to-one; F1 này không thể thay cho structural risk hoặc volume coverage. Kết luận chỉ áp dụng evaluator đã dùng, không suy rộng sang official PENGWIN evaluator. [Evidence: E4_blindspots và E4_blindspot_counts](e1_e5_evidence.json).

## 6. E5: lợi ích ranking toàn đường cong không đồng nghĩa fixed-budget utility

Pooled held-out member-mean FA-v2/presence retained M+S risk ở coverage 0,5 là **0,3780/0,5774**, nhưng ở coverage 0,8 là **0,6041/0,6044**. AURC gộp toàn coverage nên có thể tốt hơn trong khi budget review 20% gần như không cải thiện.

Supplementary paired differences `FA-v2−presence`, pooled held-out member-mean:

| Risk | Review 10%: Δ [95% CI] | Review 20%: Δ [95% CI] |
|---|---|---|
| Merge+split | −0,02653 [−0,05710; 0,01200] | −0,00027 [−0,06011; 0,05128] |
| 1−F1 | −0,00769 [−0,02047; 0,00485] | +0,00608 [−0,01401; 0,01952] |
| Missing volume | **+0,00645 [0,00173; 0,01365]** | **+0,00593 [0,00240; 0,01242]** |

Các CI M+S và 1−F1 cắt 0; missing-volume nghiêng về presence ở cả budgets. Đây là marginal exploratory intervals, không hiệu chỉnh familywise và không thay primary E3. Stage 2 review 20% có M+S retained risk **0,6146** với FA-v2 so với **0,5799** với presence, nên không gọi review utility là robust qua risk sources.

Review 40/200 bones tương ứng **34 CTs với FA-v2, 32 presence, 37 partition-v1, 33 partition-v2**. Đây là expected distinct CT count dưới cutoff ties, không phải measured reading time hoặc equal-CT-cost comparison. Không mặc định review sẽ sửa lỗi hoàn hảo.

Giữ nguyên các ví dụ E5 theo rule: `176/right_hip`, `160/sacrum`, `187/left_hip`. Ca cuối có cả ba member F1=0, missing-volume=1 nhưng disagreement=0 và M+S risk=0: lỗi bỏ sót toàn anatomy không được structural endpoint này phản ánh. Không thay bằng ví dụ thuận lợi hơn. [Hình E5](../../outputs/stage3_h1_v2/e5_supplement_v1/figures/qualitative.pdf).

Runtime E3 ghi median inference **bao gồm model load** khoảng 80,03–83,18 giây/case/model; scoring+evaluation CPU median 19,68 giây/case. Không có historical model-load/cleanup/baseline timings riêng. Không gọi 19,68 giây là overhead riêng của VI hoặc speedup so với score khác. [Evidence: E5_paired_budget, E5_fixed_budget, E5_runtime](e1_e5_evidence.json).

## 7. Claim–bằng chứng–giới hạn

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

## 8. Outline, draft và tự kiểm tra

Mini-outline cho paper: (1) setup/roles; (2) component diagnosis và count mechanism; (3) primary held-out comparison với mọi baseline; (4) controlled evaluation sensitivities; (5) fixed-budget/omission outcomes; (6) discussion và limits. Bản nháp tiếng Anh: [results_discussion_draft.md](results_discussion_draft.md). Các paragraph roles và reverse outline nằm ở cuối draft để reviewer nội bộ kiểm tra flow.

| Góc tự kiểm tra | Kết quả |
|---|---|
| Contribution | Phân tích và đánh giá có kiểm soát; không claim phát minh VI hoặc new universal metric |
| Clarity | Case/bone/model-record denominators, normalization domain và metric direction được tách rõ |
| Experimental strength | Đưa đủ component baselines, null/negative budget và severity findings; không chọn riêng hàng có lợi |
| Evaluation completeness | E1–E5 đã có; external validation và human review study chưa có; giữ là giới hạn |
| Design soundness | Không thay score/risk/split sau held-out; supplementary được gắn role; timestamped registration proof chưa xác minh |

**Agent-typed commands:** các lệnh reporting/tests dưới đây, cùng local documentation-link checks. Auditor ban đầu giả định sai cách tạo E2 signature, đã được sửa theo producer (payload E2 gồm `provenance` và `smoke_only`); experiment artifacts không đổi.

Kết quả kiểm tra cuối: **119 tests pass**, năm source config hashes khớp, 111 recorded output hashes và 142 recorded cache hashes khớp; sidecar được tái kiểm tra sau khi ghi. Các local Markdown links của tài liệu cập nhật đều tồn tại. Rà soát chéo đã sửa wording về pairwise scores so với entropy/count variance, giới hạn pooled của E4 CIs và phạm vi blindspots của internal F1. Bản nháp vẫn cần tác giả duyệt, không phải đánh giá độc lập về khả năng được venue chấp nhận.

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/exchange-e2-matched-report-mpl \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/plot_e2_matched_merge_report.py
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/audit_e1_e5_evidence.py --write
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/audit_e1_e5_evidence.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests:scripts MPLCONFIGDIR=/tmp/exchange-evidence-tests-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -m unittest discover -s tests
```

**User-typed commands:** None cần cho việc tổng hợp này. Không cần GPU hoặc submit lại experiments.

**Next step:** tác giả duyệt wording và phạm vi claims của Results/Discussion, xác minh chronology/registration evidence nếu muốn dùng ngôn ngữ preregistration, rồi tích hợp với Method/Introduction và template venue. E6/QC mới là nhánh riêng chưa triển khai ở đây.
