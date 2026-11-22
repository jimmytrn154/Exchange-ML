# Experiments để hoàn thành paper

**Mục tiêu:** kiểm tra liệu bất đồng về cách chia mảnh xương (*partition*) giúp phát hiện merge/split ngoài tín hiệu bất đồng foreground (*presence*), và xác định điều kiện đánh giá làm kết luận thay đổi. Dựa trên [plan.md](plan.md) và [methodology blueprint](research/phase1_scoping/methodology_blueprint.md).

**Đường chạy nhanh nhất:** tận dụng ba member và predictions validation hiện có; làm E1–E2 trước; chạy inference E3 một lần rồi dùng cùng outputs cho E4–E5. E6 là phần mở rộng ngoài PENGWIN để tăng sức thuyết phục, vượt phạm vi hiện tại của `plan.md`.

## 1. Tổng thể và thứ tự chạy

| Thứ tự | Experiment | Chứng minh điều gì? | Compute mới | Đầu ra cho paper |
|---|---|---|---|---|
| E1 | Phân rã score trên 40 ca validation | Partition nhỏ do chuẩn hóa hay do ensemble gần như đồng ý? — C1 | CPU, predictions có sẵn | Phân bố các thành phần và bảng ablation |
| E2 | Controlled perturbations + kiểm tra thang voxel | Score phản ứng với đúng loại lỗi; tách tác động mẫu số — C1/H4 | CPU | Figure cơ chế |
| E3 | Failure detection trên held-out | v2 có giá trị trên dữ liệu chưa dùng để thiết kế score? — C2 | GPU inference, sau đó CPU | Main table + risk–coverage curves |
| E4 | Các yếu tố gây sai lệch đánh giá | Kết luận thay đổi vì annotation, risk source, prevalence, matching thế nào? — C3 | CPU, dùng outputs E1/E3 | Bảng sensitivity |
| E5 | Qualitative + chi phí thực tế | Score bắt được và bỏ sót lỗi nào; chi phí sử dụng | Outputs E3 + thời gian đo trong lúc chạy | Panel ví dụ + runtime |
| E6 | External validation, ưu tiên kiểm tra RibFrac | Hiệu ứng có tồn tại ngoài PENGWIN? | Phụ thuộc dữ liệu/model sẵn có | Bảng external validation |

E1–E5 hoàn thành phần thực nghiệm của paper theo plan. E6 cần cho một kết luận có bằng chứng ngoài PENGWIN; pelvic/femur và một seed khác không thay thế dataset độc lập.

## 2. Phần cần chuẩn bị một lần

### Đầu vào và dữ liệu sẽ tạo

Các đường dẫn dưới đây tính từ root `Exchange-ML/`. Validation và perturbations đã có kết quả. Thư mục `heldout/` được tạo khi chạy E3.

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

**Seed đề xuất cho các bước mới:** chọn held-out `20261117`, chọn ca H4 `20261118`, bootstrap `20261119`. Giữ model seeds `31001–31003`; các seed mới không yêu cầu train lại.

Output gốc: `outputs/stage3_h1_v2/`, với ba nhánh `validation/`, `heldout/`, `perturbations/`; bảng tổng hợp đặt trong `tables/`, hình trong `figures/`. Lưu đủ các bảng dưới đây để vẽ lại hình mà không đọc lại CT.

| File trong mỗi nhánh validation/heldout | Một hàng là | Validation | Held-out 100 ca |
|---|---|---:|---:|
| `pair_components.csv` | `(case_id, bone, member_i, member_j)` | 240 | 600 |
| `scores.csv` | `(case_id, bone)`, 12 cột score | 80 | 200 |
| `model_risks.csv` | `(case_id, bone, model_id)`, ba member + Stage 2 | 320 | 800 |
| `risks.csv` | `(case_id, bone)`, member-mean và Stage 2 | 80 | 200 |
| `case_scores.csv`, `case_risks.csv` | Một ca, dùng riêng cho E4 annotation scope | 40 | 100 |

Số hàng trên là kỳ vọng khi đủ dữ liệu. Thiếu file hoặc sai hình học là lỗi cần xử lý, không được âm thầm bỏ ca. Một xương dự đoán rỗng hợp lệ vẫn có hàng score/risk; phân biệt với lỗi inference bị thay bằng mask rỗng.

### Công việc code tối thiểu

- **Hiện có:** bảng H1 v1, outputs E1/E2, checkpoint của ba member và Stage 2. Case list và config E3 đã chốt. Các volume held-out được tạo trong lượt inference E3.
- Bổ sung `VI/H(P,Q)`, scoring theo `(case, bone)`, per-member/per-bone risk, paired bootstrap và xuất các thành phần score. Tận dụng [metrics.py](../src/exchange_ml/metrics.py), [stage3_h1.py](../src/exchange_ml/stage3_h1.py), [instances.py](../src/exchange_ml/instances.py), [perturbations.py](../src/exchange_ml/perturbations.py).
- CLI v1 dùng validation. E3 có entry point riêng đọc case list đã chốt và lưu output held-out. [Lệnh E3](reports/e3_implementation.md)
- Chạy smoke trên một ca pelvic và một ca femur: hình học, ID, probability hữu hạn trong `[0,1]`, score hữu hạn, không dùng GT để tạo score. Kiểm tra identity, đổi ID, đối xứng và empty cases bằng dữ liệu nhỏ.
- Chốt cách tính trước E3: blueprint dùng AURC trapezoid trên coverage `1/n…1`, code v1 đang dùng trung bình risk theo prefix và phá hòa bằng case ID. V2 dùng quy ước blueprint, xử lý score hòa bằng kỳ vọng trên thứ tự trong nhóm hòa; tính lại **mọi baseline cùng cách**. Số AURC v1 đã công bố trong báo cáo vẫn được ghi đúng quy ước cũ.

| Nơi cần bổ sung | Chức năng cụ thể | Kiểm tra xong khi |
|---|---|---|
| `metrics.py` | Trả raw VI, joint entropy, `N`, `w`, `Q₁/Q₂` từ contingency counts | Đúng phantom E2; các giá trị finite, score chuẩn hóa trong `[0,1]` |
| `stage3_h1.py` + scoring CLI | Lọc anatomy, trả đủ 12 score và components, hỗ trợ danh sách ca | 80 score rows + 240 pair rows trên validation; GT không phải tham số scorer |
| Evaluator dùng `instances.py` | Xuất per-bone risk của cả bốn model, rồi member-mean | 320 model rows, không lấy case-level F1 thay per-bone F1 |
| Inference CLI của member/Stage 2 | Đọc case list mới, output root riêng, resume | Hai ca smoke có geometry/probability đúng; chạy lại không inference ca đã hoàn tất hợp lệ |
| Evaluation/plotting | AURC v2, paired bootstrap, ties, bảng E4 và statistical figures | Bảng/đồ thị chạy từ CSV; qualitative vẫn đọc CT/maps của ca được chọn |

Đường chạy E3 dùng lại scorer, evaluator và thống kê của E1, cùng loader ABBC của Stage 2.

**Thiết lập chung:** giữ ba member, preprocessing, 26-connectivity và ngưỡng prune prediction `1000 mm³` theo plan; GT không prune. Chọn xương có annotation bằng metadata scan type, lọc prediction bằng anatomy ID; không dùng mask GT làm ROI. Entropy dùng Stage-B probabilities trên union ROI của xương đó. Score lớn nghĩa là cần kiểm tra trước.

Các probability maps đã lưu là foreground probabilities kết hợp qua anatomy bằng voxelwise maximum. Dùng đúng các maps này và đổi ROI theo xương; không mô tả chúng là các probability channels riêng cho từng xương. Với ba map xương `P₁,P₂,P₃`, ROI chung `R_b = F₁ ∪ F₂ ∪ F₃`; mỗi cặp vẫn có `U_ij,C_ij` riêng. ROI rỗng → entropy 0 theo code hiện tại, nhưng risk vẫn tính trên GT.

Ký hiệu cho mỗi cặp: `A = |F₁ △ F₂|/|U|`, `w = |C|/|U|`, `Q₁ = VI/log|C|`, `Q₂ = VI/H(P,Q)`; `U` là hợp, `C` là giao foreground. `FA-v1 = A + wQ₁`, `FA-v2 = A + wQ₂`; lấy trung bình ba cặp. Lưu riêng `Q` và `wQ` để không gọi hai đại lượng khác nhau cùng là “partition”. Hai foreground rỗng → tổng score 0; một rỗng → 1; `|C| ≤ 1` hoặc `H(P,Q)=0` → partition 0. Hai dự đoán cùng rỗng vẫn có thể cùng sai so với GT và phải giữ trong evaluation.

## E1. Chẩn đoán trên validation — làm đầu tiên

**Đường chạy đã triển khai:** `scripts/run_e1.py --config configs/e1.json`.
E1 đã hoàn tất trên 40 ca validation và 80 đơn vị xương. [Kết quả E1](reports/e1_implementation.md)

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

## E2. Kiểm tra cơ chế — chạy song song với E1

**Đã triển khai:** `scripts/run_e2.py --config configs/e2.json`.
**Full E2 đã hoàn tất và được kiểm tra ngày 19/11/2026:** 20 ca, 40 đơn vị xương,
32/32 kiểm tra phantom, 527 response rows (452 hợp lệ), 240 cặp E1 được tái dựng.
Xem [kết quả và kiểm tra E2](reports/e2_implementation.md). Outputs ở
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

**Full đã hoàn tất trên 100 ca, 400 inference records; pooled criterion đạt.** Preparation/smoke commands giữ để tái lập, không cần chạy lại full. [Lệnh E3 và cách đọc kết quả](reports/e3_implementation.md)

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

**Full đã hoàn tất; lỗi schema bảng blindspots đã sửa với correction provenance riêng.** [Runner, kiểm tra và lệnh Slurm](reports/e4_e5_implementation.md).

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

**Full đã hoàn tất.** Runtime tách riêng chưa được E3 ghi được giữ N/A; không ước lượng thay cho số đo. Đề xuất supplementary E5 sau khi xem held-out được ghi riêng, chưa thay thiết kế bên dưới. [Chi tiết E4–E5](reports/e4_e5_implementation.md).

Supplementary CI/omission/cost full đã hoàn tất với nhãn post-heldout, outputs riêng và 10.000 bootstrap draws. [Phương pháp và audit](reports/e5_supplement_implementation.md). Giữ nguyên budgets và primary của thiết kế gốc dưới đây; kết quả không xác nhận lợi ích M+S review rõ so với presence tại 10%/20% budgets.

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

## E6. Kiểm chứng ngoài PENGWIN — phần mở rộng ưu tiên

**Lý do:** E1–E5 vẫn chỉ kiểm chứng một dataset và một backend. Nếu muốn claim vượt PENGWIN, dành nguồn lực cho external validation sau khi scorer chạy đúng; đừng chỉ tăng số ca cùng dataset.

**Kiểm tra RibFrac sớm, trong lúc làm E1–E2:** cần image–label pairs truy cập được, instance semantics phù hợp, một pipeline tạo instance maps và ít nhất ba model dự đoán trên cùng tập chưa dùng để train. **Cập nhật 21/11/2026:** strict filename/CSV inventory xác nhận đủ 420 training + 80 validation pairs; ba mask đại diện và geometry đã kiểm tra ([nguồn và semantics](reports/e6_protocol_sources.md)); model training/full inference E6 chưa chạy, nên việc đủ dữ liệu không đồng nghĩa experiment đã sẵn sàng hoặc có ETA được đo.

Nếu cần huấn luyện ba model từ đầu, ước lượng compute riêng trước khi xếp lịch. Chỉ perturb GT RibFrac có thể bổ sung kiểm tra cơ chế, **không đủ** gọi là external failure detection. Kết quả E6 được trình bày là chuyển giao score; không mặc định là chuyển giao checkpoint PENGWIN sang RibFrac.

**Nhánh chạy cụ thể khi RibFrac sẵn sàng:**

1. Ưu tiên 80 ca official validation nếu đã có đủ image–label pairs và train history của các checkpoint loại toàn bộ các ca này. Dùng cả tập, kể cả ca không có positive instance. Nếu thiếu ảnh, external run vẫn là việc chưa làm; không tự đổi sang một subset từ training rồi gọi cùng phép xác nhận.
2. Cần ba models độc lập seed, cùng backbone/preprocessing/training split của RibFrac. Không dùng ba checkpoint epoch của một run để thay ba independent members. Nếu đã có prediction cache đủ instance maps và foreground probabilities thì bỏ qua inference. Nếu chưa có cache nhưng có models: `3 × 80 = 240` inference, risk chính member-mean; model thứ tư chỉ cần nếu muốn lặp lại risk-source sensitivity.
3. Smoke ba loại ca nếu có: không positive instance, ít nhất hai instance, một instance nhỏ. Kiểm tra `label_id`/`label_code`, vùng ignore, hình học và định nghĩa target. Không biến tổn thương thành “mảnh xương nguyên khối” chỉ bằng đổi tên; nếu target khác, gọi là transfer sang fracture-instance task và nêu khác biệt.
4. Đơn vị phân tích là **case** nếu không có annotation đầy đủ theo xương. Tổng hợp score trên target foreground của dataset; không cắt theo `ANATOMY_RANGES` PENGWIN. Dùng reconstruction/cleanup của RibFrac backend đã chọn, không tune trên validation; giữ cùng outputs cho mọi score.
5. Tính 12 score E3 nếu có đúng probability maps; nếu không có, thiếu entropy baseline phải được ghi rõ và bổ sung đầu vào trước khi gọi đây là full comparison. Giữ `VI/H`, pair averaging và empty rules từ PENGWIN. Không tái chọn normalization hoặc weight trên RibFrac.
6. Với ca GT không có positive: merge/split/missing không áp dụng; ghi N/A ở structural-risk subset và báo số ca bị loại khỏi riêng endpoint này. Vẫn giữ ca đó trong đánh giá presence/false-positive: foreground rỗng cả hai → Dice=1, chỉ prediction không rỗng → Dice=0; báo false-positive instance count riêng.
7. Báo AURC và CI paired-bootstrap theo case cho structural-risk subset, cùng số ca có merge, split, missing và phân bố risk. Ít/không có merge/split thì endpoint đó không đủ thông tin; không kết luận method tốt chỉ nhờ Dice hoặc missing. Không gộp raw AURC RibFrac/PENGWIN thành một trung bình vì risk distributions khác nhau.

**Thành công của E6:** FA-v2 có paired difference có lợi trên các endpoint tương ứng, không chỉ điểm số đẹp ở dataset chính. Báo effect và CI dù âm tính; không đổi kết luận primary PENGWIN. Output `external_ribfrac/{scores,model_risks,comparison,paired_differences}.csv` là các file dự kiến cần tạo. Dataset/model readiness của E6 phải được giải quyết trước khi hứa thời gian hoàn thành nhánh này.

## 3. Cách hoàn thành nhanh

**Extension được tác giả đồng ý ngày 22/11/2026:** [Partition v3](reports/partition_v3_implementation.md) triển khai object-macro overlap descriptor, ba ridge ablations cố định và omission guard dưới cùng ngân sách CT. Development chỉ dùng 40 validation CT với nested case-grouped CV; không thay primary E3 hoặc scorer E6. Full mới phải do tác giả chạy. Fresh confirmation trên 110 candidate cases còn chờ exposure audit và protocol riêng.

**Revision sau kết quả v3, được tác giả đồng ý:** [Partition v4](reports/partition_v4_implementation.md) chỉ kiểm tra P1+G bằng pairwise linear ranking và inner selection theo AURC. Primary mới là mean outer-fold AURC; pooled OOF được giữ riêng như scale diagnostic. Đây là development sau khi xem v3, không thay primary/kết quả v3 hoặc E3, không thêm CT review policy. Full v4 dùng lại 40 validation CT từ CSV và phải do tác giả chạy.

1. **Bắt đầu ngay:** chuẩn bị scorer/evaluator v2; E1 và E2 trên CPU; kiểm tra khả dụng E6 cùng lúc.
2. **Sau E1–E2:** chốt danh sách held-out và cách đánh giá; chạy E3 trên các GPU được cấp, mỗi output chỉ có một writer. Dùng resume; không chạy inference riêng cho từng baseline.
3. **Trong khi GPU chạy:** tính score/risk theo từng ca đã hoàn thành, dựng sẵn bảng/figure E4–E5. Chưa dùng kết quả held-out từng phần để sửa method hoặc quyết định dừng lấy mẫu.
4. **Sau khi đủ ca:** tính CI, hoàn thiện main table và figures; viết kết luận theo dữ liệu. C2 không đạt thì báo giới hạn/negative result, không tiếp tục thử normalization để tìm hàng thắng.

**Phân chia việc để tránh chạy lại:** một lần đọc predictions/case → cleanup một lần → contingency counts dùng chung cho VI/log N và VI/H → xuất tất cả score. Evaluator đọc GT riêng, xuất đủ bốn model risks cùng lượt. Bootstrap/figures chạy trên CSV, không mở volume. Bắt đầu với một CPU worker đọc volume để đo RAM/I/O rồi mới tăng; không nạp cả cohort vào RAM. Trên ba GPU, ưu tiên ba member, Stage 2 dùng slot trống tiếp theo; trên bốn GPU có thể chạy bốn model đồng thời.

| Mốc hoàn thành | Điều kiện kiểm tra được | Có thể chuyển sang |
|---|---|---|
| M0 — code chạy đúng | Phantom E2a đúng số; hai ca smoke hợp lệ; không dùng GT trong scorer | E1 + E2b/c |
| M1 — hiểu tín hiệu | Đủ 80 validation units, 240 pair rows, 320 model-risk rows; có decomposition và precision projection | Chọn held-out, E3 |
| M2 — đủ held-out | 100 ca × bốn model; 200 score units, 600 pair rows, 800 model-risk rows; không ca thiếu/duplicate | Tính CI E3, hoàn tất E4–E5 |
| M3 — đủ bằng chứng paper | Main table + paired differences + perturbation + sensitivity + qualitative/runtime, kể cả kết quả âm tính | Viết Results/Discussion |
| M4 — external | Có outputs thực của E6 và diễn giải đúng target/risk | Bổ sung claim ngoài PENGWIN |

Nếu chọn 210 ca trước E3: 420 bone units, 1.260 pair rows, 1.680 model-risk rows; thay counts M2 tương ứng. Các mốc này là kiểm tra đủ dữ liệu, không phải yêu cầu mọi kết quả đều phải thắng baseline.

Ngân sách PENGWIN mới chủ yếu là **400 inference**. Theo khoảng thời gian lịch sử trong plan (100–670 giây/lượt), phép nhân cho khoảng **11–75 GPU-giờ** nếu throughput còn tương đương; đây không phải ETA hiện tại và chưa gồm CPU/I/O. Không cần train lại PENGWIN để thực hiện E1–E5.

**Đủ đầu ra để viết Results/Discussion trong phạm vi PENGWIN:** E1–E5 và supplementary đã hoàn tất. [Tổng hợp hiện tại và claim matrix](reports/e1_e5_synthesis.md) giữ các kết quả âm tính và giới hạn; [draft](reports/results_discussion_draft.md) chờ tác giả duyệt. External validation, human review utility và formal-registration evidence chưa được xác nhận ở đây. Backend fixes hỗ trợ reproducibility; không thêm claim chuyển giao khi E6 chưa có kết quả.
