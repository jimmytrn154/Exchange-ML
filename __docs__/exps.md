# Protocol và lệnh chạy

Cập nhật **22/11/2026**. Kết quả và trạng thái ở [progress.md](progress.md); ưu tiên còn lại ở [plan.md](plan.md). E1–E5, v3 và v4 đã hoàn tất: các lệnh tái lập không phải yêu cầu chạy lại. E6 fast đang triển khai production; không submit thêm writers vào root đang chạy.

## Bản đồ experiments

| Experiment | Vai trò | Config / runner | Output |
|---|---|---|---|
| E1 | Decomposition, validation benchmark | `configs/e1.json`; `scripts/run_e1.py` | `outputs/stage3_h1_v2/{validation,tables}` |
| E2 | Phantoms, GT perturbations, count replication | `configs/e2.json`; `scripts/run_e2.py` | `outputs/stage3_h1_v2/perturbations` |
| E3 | Held-out failure detection | `configs/stage3_h1_v2.json`; `scripts/run_e3.py` | `outputs/stage3_h1_v2/heldout` |
| E4–E5 | Sensitivity, qualitative, runtime | `configs/e4_e5.json`; `scripts/run_e4_e5.py` | `outputs/stage3_h1_v2/e4_e5` |
| E5 supplementary | Fixed-budget CIs, omission, figures | `configs/e5_supplement.json`; `scripts/run_e5_supplement.py` | `outputs/stage3_h1_v2/e5_supplement_v1` |
| v3 | Graph + nested ridge development | `configs/partition_v3.json`; `scripts/run_partition_v3.py` | `outputs/partition_v3_development_v1` |
| v4 | P1+G pairwise-ranking ablation | `configs/partition_v4.json`; `scripts/run_partition_v4.py` | `outputs/partition_v4_development_v1` |
| E6 | Independent RibFrac backend + fixed scores | `configs/e6.json`, `configs/ribfrac_fast.json`; `scripts/run_ribfrac_fast.py` | `/mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6-fast-v1` |

## Đầu vào và hợp đồng dữ liệu

Các đường dẫn dưới đây tính từ root `Exchange-ML/`. Validation và perturbations đã có kết quả. Thư mục `heldout/` đã được tạo khi hoàn tất E3.

| Đầu vào | Nơi đọc | Dùng cho |
|---|---|---|
| Split và initial training set | `splits/pengwin_stage0_split.json`; trajectory seed `20260920` | Chọn đúng 40 validation và pool chưa dùng |
| Scan type | Metadata pelvic/femur của dataset/split preparation | Chọn xương cần score; không suy từ prediction |
| Member instance maps | `outputs/stage3_h1/members/member_0X/validation_predictions/{case_id}.mha`, `X=1,2,3` | E1, E4 |
| Member probabilities | `outputs/stage3_h1/members/member_0X/validation_probabilities/{case_id}.mha` | Mean/p95 entropy |
| Stage 2 predictions | `outputs/stage2_backend/validation_predictions/{case_id}.mha` | Risk source thứ hai |
| CT và GT | `dataset_root` trong `configs/stage2_backend.json`; mỗi case có `image.mha`, `label.mha` | GT chỉ vào evaluator/H4; CT vào inference/qualitative |
| Checkpoint | Checkpoint inference của từng member và Stage 2 hiện có | E3; dùng cùng lựa chọn checkpoint như F9 validation |

Tên case giữ dạng chuỗi ba chữ số. Mỗi ca pelvic tạo `sacrum`, `left_hip`, `right_hip`; mỗi ca femur tạo `femur`, kể cả khi model không dự đoán được xương đó. Dải ID tương ứng: `1–50`, `51–100`, `101–150`, `151–200`.

**Seeds của protocol:** chọn held-out `20261117`, chọn ca H4 `20261118`, bootstrap `20261119`. Giữ model seeds `31001–31003`; các seed mới không yêu cầu train lại.

Output gốc: `outputs/stage3_h1_v2/`, với ba nhánh `validation/`, `heldout/`, `perturbations/`; bảng tổng hợp đặt trong `tables/`, hình trong `figures/`. Lưu đủ các bảng dưới đây để vẽ lại hình mà không đọc lại CT.

| File trong mỗi nhánh validation/heldout | Một hàng là | Validation | Held-out 100 ca |
|---|---|---:|---:|
| `pair_components.csv` | `(case_id, bone, member_i, member_j)` | 240 | 600 |
| `scores.csv` | `(case_id, bone)`, 12 cột score | 80 | 200 |
| `model_risks.csv` | `(case_id, bone, model_id)`, ba member + Stage 2 | 320 | 800 |
| `risks.csv` | `(case_id, bone)`, member-mean và Stage 2 | 80 | 200 |
| `case_scores.csv`, `case_risks.csv` | Một ca, dùng riêng cho E4 annotation scope | 40 | 100 |

Số hàng trên là kỳ vọng khi đủ dữ liệu. Thiếu file hoặc sai hình học là lỗi cần xử lý, không được âm thầm bỏ ca. Một xương dự đoán rỗng hợp lệ vẫn có hàng score/risk; phân biệt với lỗi inference bị thay bằng mask rỗng.


## Định nghĩa score và quy ước chung

**Thiết lập chung:** giữ ba member, preprocessing, 26-connectivity và ngưỡng prune prediction `1000 mm³` theo plan; GT không prune. Chọn xương có annotation bằng metadata scan type, lọc prediction bằng anatomy ID; không dùng mask GT làm ROI. Entropy dùng Stage-B probabilities trên union ROI của xương đó. Score lớn nghĩa là cần kiểm tra trước.

Các probability maps đã lưu là foreground probabilities kết hợp qua anatomy bằng voxelwise maximum. Dùng đúng các maps này và đổi ROI theo xương; không mô tả chúng là các probability channels riêng cho từng xương. Với ba map xương `P₁,P₂,P₃`, ROI chung `R_b = F₁ ∪ F₂ ∪ F₃`; mỗi cặp vẫn có `U_ij,C_ij` riêng. ROI rỗng → entropy 0 theo code hiện tại, nhưng risk vẫn tính trên GT.

Ký hiệu cho mỗi cặp: `A = |F₁ △ F₂|/|U|`, `w = |C|/|U|`, `Q₁ = VI/log|C|`, `Q₂ = VI/H(P,Q)`; `U` là hợp, `C` là giao foreground. `FA-v1 = A + wQ₁`, `FA-v2 = A + wQ₂`; lấy trung bình ba cặp. Lưu riêng `Q` và `wQ` để không gọi hai đại lượng khác nhau cùng là “partition”. Hai foreground rỗng → tổng score 0; một rỗng → 1; `|C| ≤ 1` hoặc `H(P,Q)=0` → partition 0. Hai dự đoán cùng rỗng vẫn có thể cùng sai so với GT và phải giữ trong evaluation.


## E1. Chẩn đoán trên validation — hoàn tất

**Đường chạy đã triển khai:** `scripts/run_e1.py --config configs/e1.json`.
E1 đã hoàn tất trên 40 ca validation và 80 đơn vị xương. [Kết quả E1](progress.md#e1)

**Dữ liệu:** 40 ca → 80 đơn vị `(case, bone)`; tính cả score case-level cũ để đối chiếu. Không inference lại nếu outputs F9 đầy đủ.

Đầu tiên xuất components và risk của cả bốn model theo từng xương. Dùng số mảnh **thực tế**, không giả định mọi prediction có ≤12 mảnh. So v1/v2 trên cùng đơn vị, ROI và risk. Union-NVI là đối chứng trên union, vẫn chứa khác biệt foreground/background.

**Chạy thành ba phép phân tích:**

| ID | Cố định | Thay đổi | Output và cách đọc |
|---|---|---|---|
| E1a — numerator vs denominator | Cùng cặp maps, cùng `C` | `VI/log N` → `VI/H(P,Q)` | `tables/decomposition.csv`: median/IQR/max của raw VI, H, Q, wQ và tỷ lệ raw VI≤`1e-12`; raw VI≈0 thì không kết luận chỉ do mẫu số |
| E1b — signal vs risk | Cùng 80 units, cùng member-mean risk | 12 score ở E3 | `validation/comparison.csv`: AURC/Spearman; scatter `A`–`wQ₂` tô màu theo merge+split risk; đánh giá riêng pelvic/femur |
| E1c — ngân sách xác nhận | Cùng comparison v2–baseline | Dự kiến 100 hoặc 210 ca | `tables/precision_projection.csv`: effect validation, CI half-width hiện tại và dự phóng |

Với E1a, lưu `n_shared_voxels, n_union_voxels, k_i, k_j, vi, joint_entropy, overlap_weight, presence, q_log_n, q_joint, weighted_q_log_n, weighted_q_joint, fa_v1, fa_v2`. Summaries gộp theo unit phải lấy trung bình ba cặp trước. Tỷ trọng partition là `mean(wQ)/mean(A+wQ)`; không thay bằng trung bình ba tỷ số; tổng score=0 thì tỷ trọng ghi N/A và báo số units đó.

E1c dùng `h_n ≈ h_40 × sqrt(40/n)` làm dự phóng thô với cùng tỷ lệ scan type. Lập một hàng cho mỗi baseline, không chọn hàng thuận lợi nhất. Nếu CI half-width dự phóng lớn hơn effect validation có lợi, 100 ca có thể quá thiếu chính xác; cân nhắc 210 trước E3. Effect validation≤0 không phải lý do tăng mẫu tới khi thắng. Không đặt ngưỡng “raw VI đủ tốt” từ kết quả để loại các ca khó.

**Kết quả cần có:** một bảng/figure đủ trả lời “tín hiệu partition có tồn tại, có bị mẫu số nén và có liên quan tới lỗi không?”. Nếu raw VI gần 0 và v2 không có độ phân tán, kiểm tra scoring và ensemble trước khi dùng GPU cho C2; đổi mẫu số không tự tạo thông tin.

## E2. Kiểm tra cơ chế — hoàn tất

**Đã triển khai:** `scripts/run_e2.py --config configs/e2.json`.
**Full E2 đã hoàn tất và được kiểm tra ngày 19/11/2026:** 20 ca, 40 đơn vị xương,
32/32 kiểm tra phantom, 527 response rows (452 hợp lệ), 240 cặp E1 được tái dựng.
Xem [kết quả và kiểm tra E2](progress.md#e2). Outputs ở
`outputs/stage3_h1_v2/perturbations/`. Số liệu đạt kiểm tra cơ chế; trước khi dùng
hình merge cho paper cần so cùng nhóm xương vì các mức hiện có n=29/12/4 khác nhau.

**Dữ liệu:** các phantom nhỏ để kiểm tra chính xác và 20 ca validation chọn bằng seed trước khi xem score, cân bằng pelvic/femur. Thao tác trên từng xương, giữ nguyên GT gốc; perturbations chỉ là các bản sao. Báo số đơn vị đủ điều kiện cho mỗi phép thử.

**E2a — sanity với giá trị chính xác.** Tạo foreground 64 voxel, hai mảnh bằng nhau, mỗi mảnh 32 voxel. So map ban đầu với map sau thao tác, không áp prune prediction lên phantom. Tolerance số học `1e-10`.

| Cặp so sánh | Presence | `Q₁` trên giao | `Q₂` trên giao | FA-v2 |
|---|---:|---:|---:|---:|
| Giữ nguyên hoặc hoán vị hai ID | 0 | 0 | 0 | 0 |
| Gộp hai mảnh thành một | 0 | `log(2)/log(64) = 1/6` | 1 | 1 |
| Đảo chiều so sánh trên: một mảnh → hai mảnh | 0 | 1/6 | 1 | 1 |
| Xóa một trong hai mảnh | 1/2 | 0 | 0 | 1/2 |
| Rỗng vs không rỗng | 1 | 0 theo quy ước | 0 theo quy ước | 1 |
| Cả hai rỗng | 0 | 0 | 0 | 0 |

**E2b — response trên hình học thực.** Sắp ID ca trong từng scan type, dùng RNG seed `20261118` lấy không hoàn lại 10 pelvic + 10 femur: 40 đơn vị xương. Mỗi perturbation được tạo độc lập từ GT gốc. Với các mức severity trong cùng một loại, dùng lựa chọn nhất quán như dưới đây; các loại khác nhau không cộng dồn.

| Thao tác | Thiết kế | Kiểm tra cần thấy |
|---|---|---|
| Identity / đổi ID | Giữ nguyên partition, hoán vị ID | Mọi disagreement bằng 0 |
| Merge | Gộp 1, 2, 3 cặp mảnh cùng xương nếu có đủ mảnh; giữ foreground | Presence bằng 0; partition phát hiện thay đổi |
| Split | Chia mảnh thành 2, 3, 4 phần; giữ foreground | Presence bằng 0; partition phát hiện thay đổi |
| Missing | Xóa một mảnh nhỏ, vừa hoặc lớn theo thứ hạng thể tích | Presence tăng theo thể tích mất; partition trên giao có thể bằng 0 |
| Boundary | Erode/dilate tới Dice khoảng 0,99 / 0,95 / 0,90, không đổi quan hệ instance trên giao | Partition thuần bằng 0; presence và FA tổng có thể tăng |
| Voxel replication | Nhân mọi ô của contingency table bởi `r ∈ {1,2,4,8,16}` | Raw VI và `VI/H` giữ nguyên; `VI/log(rN)` giảm |

Quy tắc tạo dữ liệu để không phải chọn bằng mắt:

- **Merge:** sắp cặp mảnh theo khoảng cách centroid tính bằng mm, tie-break theo ID; greedy lấy 1/2/3 cặp không dùng chung mảnh. Mức 3 cần ít nhất 6 mảnh. Không đủ thì đánh dấu `ineligible`, không thay bằng xương khác vì score đẹp hơn.
- **Split:** chọn mảnh lớn nhất; chọn trục có extent vật lý lớn nhất, tie-break `z,y,x`; sắp voxel theo trục rồi tọa độ, chia thành 2/3/4 nhóm gần bằng số voxel. Dùng ID còn trống trong cùng anatomy. Đây là thao tác partition có kiểm soát, không coi mọi đường chia là vết gãy thực tế.
- **Missing:** chọn mảnh nhỏ nhất, trung vị và lớn nhất theo thể tích; ties theo ID, deduplicate nếu trùng. Lưu cả phần trăm thể tích bị xóa để so đúng độ nặng.
- **Boundary:** chọn mảnh lớn nhất, dùng erosion/dilation hiện có; lưu Dice đạt được và số voxel đổi. Nếu không đạt target trong sai số `0,005`, đánh dấu `target_not_reached`, vẽ theo Dice thực tế. Không đổi nhãn giữa hai mảnh trên phần foreground chung.
- Các phép thử GT không thêm bước prune/morphology ngoài chính perturbation đang xét; nếu thêm cleanup sẽ trộn hai can thiệp. Không suy từ H4 rằng cleanup trên predictions là vô hại.

`perturbations/responses.csv`: một hàng cho mỗi `(case,bone,operation,severity)` gồm `eligible, reason, affected_ids, changed_volume_fraction, achieved_dice, presence, vi, q_log_n, q_joint, weighted_q_log_n, weighted_q_joint, fa_v1, fa_v2`. Tổng số hàng hợp lệ phụ thuộc số mảnh, phải báo `n` từng cột hình. Vẽ median/IQR qua units; nếu làm CI thì resample case, không coi các perturbations cùng ca là độc lập.

**E2c — replication.** Dùng contingency table của phantom merge và các cặp E1 có `VI>1e-12`; nhân tất cả counts, kể cả foreground mismatch counts dùng tính presence, với cùng `r`. Xuất `perturbations/replication.csv` với `r,N,VI,H,Q₁,Q₂,A,w`. Kiểm tra `VI,H,A,w,Q₂` bất biến, `Q₁(r)=VI/log(rN)`; nhóm VI=0 báo riêng.

Thí nghiệm replication dùng counts, không tạo CT lớn hay chạy resampling/inference. Nó cô lập tác động số voxel khi cấu trúc xác suất không đổi; không chứng minh model bất biến theo độ phân giải CT.

**Đầu ra:** response curves theo loại lỗi và một panel replication. Báo cả kích thước mảnh bị tác động; VI là đại lượng có trọng số thể tích, không mặc định tăng tuyến tính theo số lỗi. Không tạo entropy giả từ hard labels để đưa vào H4.

## E3. Held-out benchmark — thực nghiệm chính

**Full đã hoàn tất trên 100 ca, 400 inference records; pooled criterion đạt.** Preparation/smoke commands giữ để tái lập, không cần chạy lại full. [Lệnh E3 và cách đọc kết quả](progress.md#e3)

**Dữ liệu:** 100 ca chưa dùng để huấn luyện/validation/chẩn đoán, chọn từ 210 ca pool còn lại theo seed, 50 pelvic + 50 femur → 200 đơn vị. Kiểm tra tách biệt với dữ liệu của cả ba member và Stage 2. Giữ 50 ca test riêng. Nếu dự phóng E1 cho thấy 100 quá thiếu chính xác, quyết định dùng 210 **trước** khi xem held-out; không mở rộng dần tới khi CI có ý nghĩa.

**E3a — chọn ca:** đọc `split.pool`; trừ `initial_labeled_case_ids` của trajectory `20260920`; kiểm tra train IDs thực tế của bốn model. Dự kiến còn 105 pelvic + 105 femur. Sắp case IDs; dùng `numpy.random.default_rng(20261117)` chọn không hoàn lại 50 pelvic rồi 50 femur. Xuất danh sách `heldout/cases.csv` gồm `case_id,scan_type`; không dùng GT volume, fragment count hay model score để chọn lại ca. Nếu train history khác dự kiến, xử lý tập ứng viên trước khi chọn, không giả định đủ 210.

**Inference:** ba member × 100 ca = **300 lượt**, lưu instance maps + foreground probabilities. Để có risk Stage 2 trên held-out như blueprint yêu cầu, thêm **100 lượt Stage 2**, chỉ cần instance maps: **tổng 400 lượt**, không huấn luyện thêm. Với 210 ca, tương ứng 840 lượt. Chạy các model trên GPU được cấp độc lập, xử lý CPU theo ca đã đủ predictions.

**E3b — ma trận score:** dưới đây là đủ 12 hàng, tất cả từ cùng predictions. Dấu trung bình theo cặp luôn là trung bình ba cặp `(1,2),(1,3),(2,3)`.

| Tên trong CSV | Định nghĩa | Vai trò |
|---|---|---|
| `mean_entropy` | Mean trên `R_b` của binary entropy của probability trung bình ba member | Baseline chính |
| `p95_entropy` | Percentile 95 của cùng entropy map, interpolation `linear` | Baseline aggregation |
| `pairwise_dice` | Mean theo cặp của `1−Dice(F_i,F_j)` | Baseline mạnh về overlap |
| `presence` | Mean `A`; đúng bằng foreground Jaccard disagreement | Baseline chính |
| `union_nvi` | Mean `VI/log(n_union_voxels)` trên union, giữ label background 0 | Đối chứng trộn presence/partition |
| `count_variance` | Population variance số instance, `ddof=0`, qua ba member | Cardinality baseline |
| `hungarian` | Mean `1−sum(matched IoU)/max(K_i,K_j)` theo cặp, unmatched contributes 0 | Instance matching baseline |
| `fa_v1` | Mean `A+wQ₁` | Định nghĩa v1, tính lại cùng units v2 |
| `partition_v1` | Mean `wQ₁` | Bỏ presence khỏi v1 |
| `partition_v2` | Mean `wQ₂` | Bỏ presence khỏi v2 |
| `fa_v2` | Mean `A+wQ₂` | **Phương pháp primary** |
| `q_joint_unweighted` | Mean `Q₂` | Sensitivity bỏ trọng số giao/hợp |

Entropy: `p̄=(p₁+p₂+p₃)/3`, `h=−p̄ log p̄−(1−p̄)log(1−p̄)`, dùng log tự nhiên và `0 log 0=0`. Không lấy trung bình entropy từng member thay entropy của mean probability. Hungarian khi hai phía rỗng trả 0, chỉ một phía rỗng trả 1. Baselines giữ cách xử lý rỗng theo code, không loại unit rỗng khỏi AURC.

**E3c — risk labels:** chạy evaluator per model trước khi lấy trung bình, với cùng prediction cleanup.

| Cột risk | Công thức per model, per bone | Dùng để |
|---|---|---|
| `merge_split` | `(merge_errors + split_errors)/max(n_GT,1)` | Primary AURC |
| `one_minus_f1` | `1−instance_F1` | Độ đúng instance tổng thể |
| `missing` | `1−instance_recall` | Bỏ sót nguyên mảnh |
| `merge` | `merge_errors/max(n_GT,1)` | Tách loại lỗi |
| `split` | `split_errors/max(n_GT,1)` | Tách loại lỗi |
| `one_minus_dice` | `1−Dice(F_pred,G)` trong cùng anatomy | Semantic control |
| `missing_volume` | `1−volume(G∩F_pred)/volume(G)` | Mất một phần mảnh, dùng E4 |

`risk_member_mean=(risk_01+risk_02+risk_03)/3`; `risk_stage2` lấy từ model Stage 2. Không lấy average predictions rồi tính risk: đó là một predictor khác. Giữ thêm `n_GT, gt_volume_mm3, min_gt_fragment_volume_mm3` và chỉ số từng model để kiểm tra confounders.

Lưu ý định nghĩa code hiện có: mỗi GT chọn prediction IoU tốt nhất nếu IoU≥0,1; một prediction được nhiều GT chọn tạo merge. Split đếm các predicted IDs giao với GT, **không áp ngưỡng IoU 0,1 cho từng nhánh split**. Đây là evaluator nội bộ; diễn giải đúng thay vì gọi chung mọi lỗi là “matching IoU≥0,1”. Không đổi định nghĩa trong E3.

**E3d — thống kê và tiêu chí kết luận:**

- Risk merge+split có thể >1; không clip. Báo chất lượng segmentation và tần suất lỗi của từng model để người đọc hiểu độ khó.
- AURC thấp hơn là tốt hơn. Báo validation và held-out riêng; held-out có kết quả gộp, pelvic, femur. 200 đơn vị không phải 200 mẫu độc lập.
- 10.000 paired-bootstrap resamples **theo case**, mang theo tất cả xương; giữ phân tầng scan type và dùng cùng resamples cho các score. Báo CI 95% của chênh lệch AURC. Spearman/partial Spearman dùng GT fragment count, bone volume, min-fragment volume chỉ ở evaluation; correlation không xác định khi biến/risk hằng ghi N/A, AURC vẫn tính được với risk hằng.
- C2 theo blueprint: FA-v2 phải giảm ít nhất **0,02 AURC** so với **cả Jaccard và mean entropy**, hai CI của `v2 − baseline` nằm dưới 0. Primary dùng member-mean risk; chỉ nói kết quả bền với nguồn risk nếu Stage 2 cũng ủng hộ. Luôn so với pairwise Dice và union-NVI; thắng entropy không đủ để gọi tốt nhất.

Cách tính để các lần triển khai thống nhất:

1. Sort uncertainty tăng dần. Coverage `c_k=k/n`; risk tại `k` là mean risk của `k` units được giữ. Với một nhóm score hòa có `a` units đứng trước và tổng risk trước đó `S`, vị trí thứ `j` trong nhóm có expected prefix risk `(S+j×mean_risk_tie)/(a+j)`. Ties dựa trên score chưa làm tròn.
2. `AURC=Σ_{k=1}^{n−1}(R_k+R_{k+1})/(2n)`; không tự thêm điểm coverage=0. Random-ranking reference theo quy ước này là `(1−1/n)×mean(risk)`. Với `n<2`, AURC ghi N/A.
3. Mỗi bootstrap draw lấy có hoàn lại 50 pelvic + 50 femur case IDs cho held-out 100; validation dùng 20+20, held-out 210 dùng 105+105. Phân tích từng scan type chỉ resample nhóm đó. Mang toàn bộ bones và giữ multiplicity nếu case được lấy nhiều lần; không dùng dictionary theo case ID làm mất bản sao. Cùng draw cho mọi score và cả hai risk sources; CI percentile 2,5/97,5 của paired difference, không lấy hiệu hai CI riêng.
4. Partial Spearman dùng **rank residualization**: average-rank score, risk và ba confounders; hồi quy rank score/risk trên intercept + ranks confounders; Pearson của hai residuals. Đây khác hàm hiện tại residualize giá trị thô rồi Spearman. Áp dụng cùng cách cho mọi hàng v2; hằng hoặc rank-deficient không ước lượng được thì ghi N/A.
5. Pooled analysis cho mỗi bone trọng số bằng nhau nên pelvic đóng góp ba hàng/case; luôn đặt pelvic/femur cạnh pooled. Bootstrap theo case xử lý phụ thuộc, không tự biến pooled thành case-weighted metric.

`heldout/comparison.csv`: `score,risk_source,risk,stratum,n_cases,n_units,aurc,rho,partial_rho`. `heldout/paired_differences.csv`: `score,baseline,risk_source,risk,stratum,delta_aurc,ci_low,ci_high`. So sánh chính là hai dòng FA-v2 với presence/mean entropy; các dòng còn lại là phân tích secondary, không quảng bá mọi CI như các kiểm định primary độc lập.

**Main table:** 12 score × `{merge+split AURC, 1−F1 AURC}` cho validation và held-out; cột Δ và CI so với presence/entropy cho FA-v2. Bảng bổ sung chứa Stage 2, missing/merge/split, scan-type strata và partial correlation. Figure risk–coverage chỉ cần FA-v2, partition-v2, presence, mean entropy, union-NVI, random; thêm pairwise Dice nếu khác presence đáng kể.

**Đầu ra:** main comparison table, paired differences và risk–coverage curves có random-ranking reference. Nếu chỉ partition-only thắng, trình bày đúng kết quả của thành phần đó; không đổi primary từ FA-v2 sang hàng tốt nhất sau khi xem held-out.

## E4. Tách ảnh hưởng của cách đánh giá — cùng outputs E1/E3

**Full đã hoàn tất; lỗi schema bảng blindspots đã sửa với correction provenance riêng.** [Runner, kiểm tra và lệnh Slurm](progress.md#e4).

| Yếu tố | So sánh có kiểm soát | Đại lượng báo cáo |
|---|---|---|
| Chuẩn hóa | `log N` vs joint entropy, cùng miền và risk | ΔAURC, tỷ trọng partition, thay đổi thứ hạng |
| Annotation scope | Whole-case ROI vs chỉ các anatomy được annotate; cùng case-level risk. Sau đó mới báo phân tích per-bone riêng | Score từ xương ngoài phạm vi annotation và ΔAURC; không quy mọi voxel ngoài GT thành xương không annotate |
| Risk source | Member-mean vs Stage 2, giữ score và đơn vị cố định | ΔAURC và thứ hạng baseline theo từng nguồn |
| Failure prevalence | “Có merge/split” vs nhóm 25% risk cao nhất | Top-10%/20% enrichment; prevalence và ceiling `1/p`. Nếu hòa ở ngưỡng quartile, dùng trọng số đồng hạng để đạt 25%, không chọn tùy ý theo case ID |
| Matching bỏ qua mất một phần mảnh | Giữ F1 theo IoU≥0,1; bổ sung `missing_volume = 1−volume(G∩F_pred)/volume(G)` theo cùng xương | Số đơn vị F1=1 nhưng mất >10% GT volume; AURC theo risk thể tích và ví dụ trực quan |

Không suy ra nguyên nhân bằng cách đồng thời đổi ROI, normalization và risk source. Phân tích ngoài GT chỉ được dùng retrospectively; không đưa GT vào scorer. Sensitivity này phải được xác định trước khi xem E3.

**E4a — normalization × annotation scope, thiết kế 2×2.** Trên mỗi split, tạo bốn biến thể case-level: `{VI/log N, VI/H} × {mọi anatomy dự đoán, chỉ anatomy có annotation}`. Risk case giữ cố định: mỗi model lấy tổng merge+split trên các xương được annotate chia tổng GT instance count của các xương đó, rồi mean ba model. So từng cặp chỉ khác một yếu tố. Báo ΔAURC của normalization ở mỗi scope và ΔAURC của scope ở mỗi normalization. Các score per-bone E3 là phân tích riêng, không dùng để suy ra hiệu ứng của riêng scope.

Định lượng ngoài annotation bằng `fraction_outside_scope = |D_all ∩ O|/|D_all|`, trung bình theo cặp; `D_all` là foreground symmetric difference, `O` là vùng ít nhất một prediction mang anatomy ID không được annotate trong scan type đó. Mẫu số 0 ghi N/A. Đây là **vị trí bất đồng được prediction gán vào anatomy ngoài scope**, không phải ground-truth anatomy. Giữ riêng số voxel ngoài GT foreground; số đó còn chứa sai biên/false positives, không chứng minh tất cả là xương chưa annotate.

**E4b — risk-source sensitivity.** Với cùng từng score per-bone, lập hai cột AURC cho member-mean và Stage 2; báo rank của score ở mỗi cột và paired differences v2–baseline. Stage 2 khác seed nhưng cùng họ backend/dữ liệu; kết quả ở đây kiểm tra dependence vào risk source, chưa phải cross-backend validation.

**E4c — enrichment khi gần mọi ca đều có lỗi.** Giữ cùng ranking của score. Tính riêng label `r_MS>0` và label nhóm severity cao nhất. Với `n` units, nhóm severity có tổng trọng số `q=0,25n`; nếu có `a` units có risk cao hơn mức cutoff và `t` units bằng cutoff, mỗi unit ở cutoff nhận nhãn mềm `(q−a)/t`. Nhóm được review lấy `k=ceil(0,1n)` hoặc `ceil(0,2n)` units score cao nhất; nếu hòa ở cutoff score, dùng trọng số chọn đồng đều tương tự. `Enrichment = mean(label trong nhóm được chọn)/mean(label toàn tập)`. Khi prevalence=0 hoặc mọi risk bằng nhau, ghi không đủ thông tin; không diễn giải top quartile là clinical failure.

**E4d — thiếu thể tích dù F1 hoàn hảo.** Tính `missing_volume` cho từng `(case,bone,model)` trước khi aggregate. Xuất mọi hàng `F1≥1−1e-12` và `missing_volume>0,10`; báo số ca và số xương riêng, không coi bốn model cùng ca là bốn bệnh nhân. Thêm bảng theo từng GT fragment: `gt_id, matched_pred_id, matched_iou, gt_volume_mm3, gt_fraction_covered`. Coverage của fragment đo phần GT được **bất kỳ** prediction foreground cùng anatomy phủ, không chỉ prediction được match; phân biệt volume loss với chia/gộp ID.

**File cần xuất:** `tables/normalization_scope.csv`, `tables/risk_source.csv`, `tables/enrichment.csv`, `tables/f1_volume_blindspots.csv`. Mỗi file có `split` và `stratum`; không trộn validation với held-out để tăng cỡ mẫu sau khi đã dùng validation chọn method.

**Đầu ra:** một bảng compact cho C3; toàn bộ được tính từ predictions có sẵn, không cần thêm inference.

## E5. Ví dụ thành công, thất bại và chi phí

**Full đã hoàn tất.** Runtime tách riêng chưa được E3 ghi được giữ N/A; không ước lượng thay cho số đo. Đề xuất supplementary E5 sau khi xem held-out được ghi riêng, chưa thay thiết kế bên dưới. [Chi tiết E4–E5](progress.md#e4).

Supplementary CI/omission/cost full đã hoàn tất với nhãn post-heldout, outputs riêng và 10.000 bootstrap draws. [Phương pháp và audit](progress.md#e5). Giữ nguyên budgets và primary của thiết kế gốc dưới đây; kết quả không xác nhận lợi ích M+S review rõ so với presence tại 10%/20% budgets.

**E5a — chọn hình bằng quy tắc cụ thể.** Dùng percentile rank trung bình khi hòa, tính trong từng scan type trên held-out; score càng cao percentile càng lớn. Chỉ chọn từ các ca inference hoàn tất, không loại ca vì hình khó nhìn.

| Panel | Điều kiện chọn | Nếu có nhiều ứng viên |
|---|---|---|
| Partition bắt merge/split | `r_MS>0`, percentile partition-v2≥0,75; presence và entropy đều≤0,50 | Max `rank(partition)−max(rank(presence),rank(entropy))` |
| Presence bắt missing | `r_missing>0`, presence≥0,75, partition-v2≤0,50 | Max missing-volume risk |
| Ensemble cùng sai | Cả ba member có `1−F1>0`; mean `1−F1≥0,20`; FA-v2≤0,25 | Max mean `1−F1` |

Tie-break cuối theo `(case_id,bone)` tăng dần. Ngưỡng rank ở bảng là percentile, trừ các ngưỡng risk ghi rõ. Nếu không có ứng viên thì ghi “không có ca thỏa điều kiện”, không nới cutoff sau khi xem hình. Panel thứ ba gọi là “cùng sai với bất đồng thấp”, không mặc định cả ba mắc đúng cùng một loại lỗi.

Xuất `heldout/qualitative_candidates.csv` trước khi vẽ. Với ca được chọn, dùng chung crop và lát cắt đi qua lỗi; ghi tọa độ lát cắt. Align màu instance với GT bằng overlap chỉ cho visualization, không dùng mapping đó trong scoring. Hình gồm CT, GT, ba predictions, score và per-model risk; nếu lỗi merge/split không nhìn rõ trên một lát, dùng hai lát hoặc 3D inset.

**E5b — ngân sách review.** Với mỗi score và source risk, giữ `floor(0,8n)` hoặc `floor(0,9n)` units có score thấp nhất; xử lý ties như risk–coverage. Xuất `coverage, reviewed_units, expected_reviewed_cases, retained_mean_risk, delta_vs_random, delta_vs_presence`; random reference ở coverage cố định là mean risk toàn tập. Ngân sách tính theo xương, nên không gọi 20% bones là 20% CT scans. Nếu cutoff không cắt nhóm hòa, số ca review là số case IDs riêng biệt. Nếu chọn `q` trong `t` units hòa, một ca chưa được chọn chắc chắn, có `t_c` units trong nhóm, được review với xác suất `1−choose(t−t_c,q)/choose(t,q)`; cộng xác suất để có số ca kỳ vọng.

**E5c — runtime.** Bắt đầu đo trong các lượt E3 vốn phải chạy, không tạo thêm job timing. Với 10 ca đầu mỗi model, lưu `case_id,scan_type,n_voxels,model_load_seconds,inference_seconds,write_seconds,peak_vram`; scoring lưu `read_seconds,cleanup_seconds,components_seconds,baselines_seconds,peak_rss`. Báo median/IQR theo scan type và thời gian model load riêng. Scoring CPU đo wall-clock thực, không quy inference time thành overhead của VI.

**File cần xuất:** `figures/qualitative.pdf`, `tables/review_budget.csv`, `tables/runtime.csv`. Định dạng hình cuối là PDF/SVG hoặc PNG đủ độ phân giải, có legend và chiều “score cao = bất đồng cao”.

**Đầu ra:** một panel 2–3 ví dụ và vài số runtime/review budget. Không mở benchmark kiến trúc hoặc train thêm model chỉ để bổ sung hình.

## E6. RibFrac external validation và performance amendment

### Dữ liệu và backend

Chuyển giao fixed disagreement scores sang **rib-fracture lesion instances**, không phải whole-rib/bone-fragment segmentation và không chuyển PENGWIN checkpoints. Giữ mọi nonzero GT instance ID, kể cả `label_code=-1`; −1 chỉ loại khỏi fracture-type classification. Không dùng ID anatomy PENGWIN cho RibFrac.

420 official training subjects tạo split nội bộ cố định **336 train / 84 validation** (seed `20261121`); **80 official validation subjects (IDs421–500)** chỉ dùng external evaluation. Không đưa external vào fingerprint, planner, internal validation, threshold tuning hoặc chọn checkpoint. Không dùng 160 test images.

Backend binary nnU-Net 3D full-resolution `nnunetv2==2.5.1`, train từ đầu với ba seeds **32001/32002/32003**; cùng split, preprocessing và architecture plan. Budget **1.000 epochs**, chọn **final checkpoint**. Native-grid softmax threshold **0,5**, **26-connectivity**, không prune component theo ngưỡng thể tích PENGWIN. Touching lesions có thể merge và disconnected lesions có thể split; không sửa bằng GT.

12 fixed E3 scores được tính case-wise từ predictions/probabilities trước khi đọc GT. Hungarian dùng generic IDs, sparse assignment tương đương maximum summed IoU. Primary risk là mean của risk từng model; giữ internal matching **IoU≥0,1**, không gọi là official RibFrac FROC. Zero-GT cases có structural/missing endpoints N/A, vẫn có F1/Dice/false-positive evaluation. Báo denominators, exact ties và **10.000 paired case-bootstrap draws**; endpoint constant-risk ghi uninformative. Không tạo success threshold mới sau kết quả hoặc gộp AURC với PENGWIN.

### Revision batch/cache đã được tác giả chọn

- Root gốc: `/mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6`; root fast riêng như bảng trên, không ghi đè nguồn.
- Baseline batch2: 250 train/50 validation iterations, 500/100 patches mỗi epoch. Batch5: 100/20 iterations, 500/100 patches. Batch6: 83/17 iterations, 498/102 patches.
- Giữ patch/spacing/architecture, 1.000 epochs và LR schedule theo epoch; không tự scale LR. Optimizer updates, batch Dice và foreground-oversampling rounding thay đổi nên đây là explicit amendment, không phải optimization-equivalent run.
- NPY mmap cache verify source hashes/values/dtypes; không giữ mở toàn cohort. Cache bốn workers, augmentation một process theo RNG contract. Cùng environment, AMP; cuDNN benchmark bật cùng deterministic flag.
- Checkpoint mỗi năm epochs. Handoff verify nguồn, owner, identities và writer locks trước khi dừng chain cũ. `handoff.json`/`warmstarts/*/source.json` ghi source hashes, seeds và starting epochs. Không coi old/new roots là thêm independent seeds.
- Complete final run cần **3 distinct trained final checkpoints, 80 external cases, 240 inference records**, complete manifests và verified hashes. Smoke giả lập không phải external evidence.

## Partition development v3/v4

V3 dùng graph-degree disagreement G cùng P1 và presence, nested grouped OOF **5 outer / 3 inner folds**, CT cùng fold và stratified scan type; final family ridge(P1,G,presence), target `log1p(member_mean_merge_split)`. Chọn alpha theo inner grouped MSE, final family cố định. GT không tạo predictor features; risk labels lấy frozen E1 CSV.

V4 kiểm tra đúng **P1+G không presence** bằng linear pairwise logistic ranking: `mean_pairs log(1+exp(−t_ij (z_i−z_j)^T beta)) + alpha/2 ||beta||²`. Chuẩn hóa chỉ fit trên training portion. Giữ 5×3 folds và seed `20261122`; alpha grid `[0.001,0.01,0.1,1.0]`, chọn mean inner-fold AURC; exact ties chọn alpha lớn nhất. Primary là arithmetic mean outer-fold bone AURC; pooled OOF là secondary scale-mixing diagnostic.

10.000 paired CT-bootstrap draws; v4 resample trong fixed fold và scan type, giữ mọi bone của CT cùng nhau. CI conditional trên fixed models/OOF scores, không refit. V3/v4 tái dùng validation sau E3, không phải independent confirmation, không đổi primary E3. Chi tiết config/code và artifacts được giữ nguyên.

## Lệnh kiểm tra nhẹ

Các lệnh dưới chỉ audit/check/tests; agent được chạy. Dùng đúng project environment để tránh API khác của nnU-Net trong base environment.

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/audit_e1_e5_evidence.py
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/run_ribfrac_fast.py check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. MPLCONFIGDIR=/tmp/exchange-ml-mpl \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -m unittest discover -s tests -v
```

`audit_e1_e5_evidence.py --write` chỉ tái tạo reporting sidecar từ các sources đã hash-verify khi chủ ý cập nhật báo cáo; không chạy inference, scoring hay bootstrap experiments. Sidecar/hình reporting ở `outputs/stage3_h1_v2/reporting/`; không ghi thêm files trong `__docs__`.

## Lệnh theo dõi E6 đang chạy

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
squeue -u "$USER"
tail -n 20 outputs/logs/ribfrac-fast-train-2923_0.log
tail -n 20 outputs/logs/ribfrac-fast-train-2923_1.log
```

Job IDs là snapshot chain ngày 22/11/2026, không phải IDs cố định cho các lần resume. Xem scheduler, log và manifests cùng nhau trước khi kết luận trạng thái hoặc ETA. Timeout/failure có thể chặn downstream `afterok`.

## Commands full do tác giả chạy khi cần tái lập

**Không cần chạy lại E1–E5/v3/v4 đã complete. Không submit full E6 fast khi chain hiện tại còn writers.** Nếu cần chạy lại/resume, xác định jobs cũ đã dừng, root/identity/checkpoint phù hợp và dependencies trước; giữ một writer/member. Các commands dưới là entry points để tái lập, không phải commands đã được agent chạy trong lần gộp docs.

E6 fast chain: GPU probe → CPU cache → CPU handoff → GPU train array → GPU infer array → CPU evaluation. Một GPU/task, tối đa hai tasks đồng thời. Launcher có bước cancellation của chain gốc đã cấu hình, nên phải kiểm tra owner/job identity trước khi tái dùng.

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash scripts/submit_ribfrac_fast.sh full
```

V3/v4 chỉ dùng CPU (cluster partition có thể tên `gpu`, không request GPU):

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash scripts/submit_partition_v3.sh full
bash scripts/submit_partition_v4.sh full
```

E3 reproduction dùng một Slurm GPU/model; bốn model, tối đa hai tasks đồng thời; CPU evaluation chỉ submit sau inference thành công:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mkdir -p outputs/logs
full=$(sbatch --parsable --array=0-3%2 scripts/e3_infer.sbatch full) &&
  sbatch --dependency="afterok:${full%%;*}" scripts/e3_evaluate.sbatch full
```

E4/E5 và supplementary là CPU-only:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
sbatch scripts/e4_e5.sbatch full
sbatch scripts/e5_supplement.sbatch full
```

Các transcripts/lệnh chuẩn bị backend lịch sử ở archive, không phải runbook hiện tại. Không chạy chúng vào roots frozen hoặc roots đang có writer.

## Follow-up sau rà soát: Slurm CPU

**Cập nhật23/11/2026 — count/evaluator full đã complete và hash-audited; xem progress.md để đọc kết quả.** Tác giả đã đồng ý triển khai sau rà soát. Toàn bộ follow-up dưới dùng CPU, không train/infer thêm, không đổi source/config/checkpoints của E6 đang chạy. Tên partition Slurm là `gpu` nhưng scripts không request GPU.

| Nhánh | Config / entry | Output | Slurm resources |
|---|---|---|---|
| Reproducibility + exposure preflight | `scripts/audit_reproducibility.py` | `outputs/research_followup_v1/preflight/` | 2 CPUs, 4 GiB, 20 phút |
| P1+count ablation | `configs/count_rank.json`; `scripts/run_count_rank.py` | `outputs/count_ablation_development_v1/` | 4 CPUs, 8 GiB, 2 giờ |
| Evaluator sensitivity | `configs/evaluator_sensitivity.json`; `scripts/run_evaluator_sensitivity.py` | `outputs/evaluator_sensitivity_v1/` | 4 CPUs, 32 GiB, 12 giờ |
| E6 positive/negative report | `scripts/report_ribfrac_cohorts.py` | `outputs/research_followup_v1/ribfrac_cohorts/` | 2 CPUs, 4 GiB, 15 phút; sau E6 |

Time limits là giới hạn tài nguyên, không phải ETA. Scheduler quyết định lúc bắt đầu.

### P1+count: đúng một ablation đã chọn

Candidate `pairwise_p1_count` dùng **P1 và absolute predicted instance count mean**. Giữ v4 grouped 5 outer/3 inner folds, seed `20261122`, mean pairwise-logistic objective, alpha grid `[.001,.01,.1,1]` chọn bằng inner-fold AURC, và primary mean outer-fold AURC. Predictors không có GT count, GT volume, anatomy labels hoặc held-out outcomes. Tái sử dụng frozen validation CSVs; không đọc volumes.

Controls gồm đủ 12 scores gốc, G, count-only, frozen v4 OOF candidate, tổng 16 scores cùng candidate mới. Runner xác minh E1/v3/v4 manifests, code/config/source hashes và identical outer folds trước khi fit. Báo paired CT-bootstrap 10.000 draws, pelvic/femur/per-anatomy; pooled OOF chỉ là scale diagnostic. GT-count bins `1`, `2–5`, `≥6` chỉ vào descriptive complexity table, không vào fitting hoặc chọn alpha.

Model lưu native solver status. Nếu L-BFGS-B báo line-search failure ở nghiệm gần dừng, count implementation tự tính lại analytic gradient và chỉ chấp nhận stationarity ≤`1e-7`; nonfinite/nonstationary nghiệm vẫn bị từ chối. Đây là kiểm tra số học của code mới, không sửa thuật toán/control v4 cũ.

### Evaluator sensitivity: giữ đầy đủ endpoint cũ

Đọc predictions đã lưu cho **40 validation +100 held-out CTs**, từng ca/từng model, GT chỉ vào evaluator. Không đọc probability maps để tạo score mới: tái dùng đủ 12 scores frozen. Reconcile per-model original risks trước khi tính supplementary endpoints; geometry, prediction cleanup, model receipts và case membership phải khớp.

- One-to-one matching: maximum-cardinality admissible IoU≥0,1, tie-break utility theo IoU; báo `1−F1` riêng.
- Significant-overlap merge/split: cạnh phải có `intersection/max(GT volume,pred volume)≥tau`, báo **cả tau=0,01 và 0,1**. Denominator max đòi hỏi coverage ở cả hai phía; dùng min sẽ không lọc được satellite một voxel nằm trong GT lớn. Rule có thể lọc cả split thật của mảnh rất nhỏ; không coi thresholds là clinical severity đã được xác nhận.
- Original M+S, original `1−F1` và missing-volume giữ cùng bảng. Báo cả member-mean và Stage 2, paired CT-cluster bootstrap 10.000 draws.
- Equal-CT budgets 10%/20%: P1, guard+P1, FA-v2, presence. Guard dùng expected anatomy và all-member-empty flags prediction-only, không dùng GT để chọn CT. Ties lấy kỳ vọng; nếu flagged CTs vượt ngân sách, báo infeasible. Omission positives/undefined bootstrap draws và CI availability được ghi rõ; không tự bỏ các draws bất lợi.

Cache từng ca có source hashes, root/run identity, payload digest và incremental manifest receipt. Resume kiểm tra toàn bộ trước khi reuse; source/cache drift bị từ chối. Đây là sensitivity hậu E3 trên cases đã xem, **không phải fresh confirmation** và không thay primary E3.

### Command full do tác giả chạy

Một command submit **CPU preflight → hai synthetic smokes → hai full jobs**, mỗi full chỉ chạy sau smoke tương ứng thành công:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash scripts/submit_research_followup.sh full
```

Không thêm E6 cancellation/restart vào chain này. `check` chỉ submit preflight; `smoke` submit preflight và hai synthetic jobs, không submit full.

Có thể submit từng nhánh độc lập, mỗi launcher tự tạo synthetic smoke → full:

```bash
bash scripts/submit_count_rank.sh full
bash scripts/submit_evaluator_sensitivity.sh full
```

Chọn master launcher **hoặc** launchers riêng để tránh duplicate submissions. Locks ngăn concurrent writers cùng root. Rerun cùng identity verify complete hashes rồi thoát; nếu lỗi, xem log/cause trước khi resubmit. Không đổi code/config rồi resume cùng output identity.

Logs: `outputs/logs/research-preflight-*.log`, `count-ablation-*.log`, `evaluator-sensitivity-*.log`. Full receipts: `outputs/count_ablation_development_v1/rank_manifest.json` và `outputs/evaluator_sensitivity_v1/sensitivity_manifest.json`; cần `status=complete` và hash checks, không kết luận từ job submission thành công.

### Báo cáo RibFrac sau E6 evaluation

Đọc `e6_manifest.json` complete, verified `risks.csv`, `model_risks.csv`, `summary.json`; yêu cầu đủ 80 official external cases/ba model rows mỗi ca và 60 positive/20 negative subjects. Báo FP instances/CT, tỷ lệ CT có FP, F1/Dice và structural eligibility theo cohort/model. Member-mean là trung bình ba model, không phải consensus hoặc 240 patients. Negative structural risks giữ undefined. Không tính official FROC, không tạo CI mới hay thay matching/threshold.

Snapshot đã kiểm tra ngày 22/11: E6 evaluation job **2925** vẫn pending dependency. Để queue report sau đúng evaluation job hiện tại:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
sbatch --dependency=afterok:2925 scripts/ribfrac_cohorts.sbatch
```

Nếu chain E6 đã thay, dùng evaluation job ID mới. Reporter fail-closed nếu source chưa complete hoặc hashes/counts sai; không ghi vào external E6 root.

## Dual-channel CT triage sau các ablations âm tính

**23/11/2026: synthetic smoke và full đã complete (Slurm2986/2987).** Một policy cố định tách P1 (structural) và presence (foreground), không fit thêm weight theo GT. Đây là thử nghiệm follow-up hậu nghiệm trên cohorts đã xem, không phải fresh confirmation hoặc primary E3 mới.

- Config `configs/triage_policy.json`; runner `scripts/run_triage_policy.py`; module `src/exchange_ml/triage_policy.py`.
- Mỗi channel đổi sang midrank ECDF theo từng anatomy, tham chiếu prediction scores của validation. Validation dùng 5 CT-grouped folds seed20261122; mỗi CT chỉ được hiệu chỉnh bằng các CT ngoài fold. Held-out dùng toàn bộ40 validation làm reference. Không có outcome/GT count trong calibration.
- Candidate `guard_dual_channel`: lấy max hai channel percentiles rồi max theo bone trong CT, giữ expected-anatomy all-members-empty guard. Baseline chính `guard_p1` giữ **cùng guard**. Ten policies có raw/cali/single-channel/non-guarded controls để tách hiệu ứng calibration, combination và guard.
- Follow-up target được ghi trong config: pooled member-mean missing-volume retained risk ở20%CT review, candidate−guardP1. Cùng bảng báo10%budget, original/significant M+S, one-to-one F1, Stage2 và pelvic/femur. Không dùng target mới để thay primary cũ; không chọn endpoint/threshold thắng sau run.
- Equal-cost1CT/1review, exact cutoff ties, 10.000 paired CT bootstrap draws seed20261123. CI conditional trên fixed calibration/policy; không refit trong bootstrap. Undefined omission draws/guard-infeasible budgets được báo, không bỏ draws để tạo CI thuận lợi. Không giả định review sửa lỗi hoàn hảo.
- Source là verified `outputs/evaluator_sensitivity_v1/risks.csv` cùng parent manifest/output/cache/source hashes; không đọc volumes, train hoặc infer mới, không dùng110candidates. Không áp anatomical guard vào RibFrac.

**User-typed Slurm command:**

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash scripts/submit_triage_policy.sh full
```

Launcher submit CPU synthetic smoke rồi full `afterok`; **4 CPUs,8GiB,2giờ time limit**, không request GPU. `check` chỉ submit metadata preflight; `smoke` không submit full. Output `outputs/triage_policy_development_v1/`, receipt `triage_manifest.json`; logs `outputs/logs/dual-triage-<job_id>.log`. Time limit không phải ETA. Full command chưa được agent chạy.

Full policy đã complete: target missing-volume tại20%CT trên100heldout có candidate−guardP1 **+0,00109 [−0,02321; +0,01281]**, chưa xác nhận lợi ích. Giữ output và negative finding; không đưa policy thành contribution. E6 chưa có external evaluation hoàn chỉnh.

## Tái tạo manuscript và overview thật (CPU)

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B \
  paper/build_paper.py --compile
```

Thêm `--refresh-overview` để đọc lại một CT development và ba predictions đã lưu, rồi vẽ lại overview. Default kiểm tra hashes/manifests và tái tạo hai bảng, Markdown, PDF; không train/infer hoặc chạy lại bootstrap. File build ở `paper/build/` được ignore; provenance ở `paper/evidence/`. Chi tiết trong `paper/README.md`.
