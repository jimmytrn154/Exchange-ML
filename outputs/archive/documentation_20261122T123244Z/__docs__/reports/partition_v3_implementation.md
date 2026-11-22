# Partition v3: object-macro descriptor và structural risk head

**22/11/2026 — implementation của hướng đã được tác giả đồng ý.** Đây là nhánh development sau khi đã xem E3, có version/output riêng. Full job 2908 đã hoàn tất trên 40 CT/80 bones trong 6 phút 57 giây, exit 0. Candidate pooled M+S AURC 0,4182 kém raw P1 0,2706; G riêng 0,2690 chưa có lợi ích rõ so với P1. [Chẩn đoán sau full](../research/partition_improvement_2026/assessment_after_v3.md) giải thích thêm cross-fold score shifts và giới hạn structural/coverage. [Đề xuất và cơ sở khoa học](../research/partition_improvement_2026/proposal.md), [thiết kế](../research/partition_improvement_2026/design.md) được giữ như hồ sơ trước implementation; không sửa trạng thái lịch sử trong các tài liệu đó.

## Thay đổi thuật toán

Với overlap counts `n_ij` trên shared foreground của hai predictions, đặt `d_i = (sum_j n_ij)^2 / sum_j n_ij^2`. Descriptor `G` là trung bình đối xứng của macro-mean `d_i−1` theo từng hướng. Mỗi bone lấy trung bình cả ba member pairs. ID tùy ý không ảnh hưởng kết quả; không ép matching một–một. Không có shared support thì `G=0`, đồng thời xuất support diagnostics. Zero không chứng minh prediction đúng.

Một instance chia đều thành `k` counterparts cho `G=(k−1)/2`. Macro averaging tránh để voxel mass của mảnh lớn quyết định mọi đóng góp. Split rất mất cân bằng vẫn bị giảm trọng số, nhiễu và boundary jitter vẫn có thể tăng score. Đây là descriptor thử nghiệm, không phải xác suất hoặc một metric mới đã được xác nhận. `phantom_checks.csv` giữ cả các hạn chế đó, với raw VI, P1/P2, macro conditional entropy, Hungarian và count variance làm controls. Toy size-swap/tiny-split là **unpruned**; không được diễn giải chúng như tổn thương thực đã qua cleanup.

Ba head cố định là `P1`, `P1+A`, `P1+G+A`; candidate cuối luôn là `ridge_p1_g_a`. Không chọn family có OOF đẹp nhất. Ridge tối thiểu hóa tổng squared residual của `log1p(member_mean_merge_split)` cộng `alpha*||coefficient||²`; intercept không regularize. Score triển khai là predicted log-risk dùng để xếp hạng, có thể âm hoặc lớn hơn một. Không clip thành xác suất.

Mặc định: 5 outer folds, 3 inner folds, grouped theo CT và stratified scan type; seed `20261122`; alpha `[0.1,1,10,100]` chọn bằng inner bone-macro MSE, ties chọn alpha nhỏ hơn. Mọi xương của một CT nằm cùng fold; mean/std chỉ fit trên training fold. Final model fit 40 CT sau khi chọn alpha bằng grouped inner CV trên development. Final model không dùng để tính performance trên chính training data: bảng evaluation chỉ dùng OOF predictions.

## Dữ liệu và giới hạn claim

- Chỉ 40 validation CT/80 bones của E1 vào development. Giữ nguyên native geometry, 26-connectivity, prune `1000 mm³`, anatomy ranges và empty rules cũ.
- Đọc ba saved prediction maps/CT. Không mở GT volumes, probabilities, E3 held-out scores hay RibFrac outcomes. Nhãn supervised/evaluation lấy từ CSV risk E1 đã khóa hash.
- Features vào head chỉ có `partition_v1`, `graph_degree`, `presence`. Predicted count/volume và support flags chỉ xuất để audit. Không dùng GT count/volume hay scan type làm predictors.
- So 12 frozen scores, standalone `G` và cả ba OOF heads, với member-mean/Stage 2 risks, pooled/pelvic/femur. Main extension contrast là candidate trừ **raw P1** trên pooled member-mean M+S AURC; negative delta có lợi. Giữ P2, FA-v2, Hungarian/count controls và kết quả bất lợi.
- 10.000 paired bootstrap draws resample toàn CT, giữ các xương đi cùng và scan-type strata. CI **conditional trên fixed OOF predictions**; không refit training trong bootstrap, không bao phủ toàn bộ uncertainty của quy trình học. Mọi so sánh là development, chưa phải confirmation.

Nested CV không loại bỏ việc ý tưởng đã hình thành từ chẩn đoán E1/E3. Cohort development này vẫn đã được quan sát; OOF không tạo ra một test set độc lập.

## CT review policy

Ngân sách cố định `ceil(10%*n_CT)` hoặc `ceil(20%*n_CT)`; CT score là maximum bone score. Mỗi CT tính một lần. Ties được chọn đồng đều không hoàn lại. Bone-macro giữ đúng kỳ vọng của tỷ số tổng risk/tổng bones, gồm trường hợp hòa giữa CT pelvic 3 bones và femur 1 bone; không thay bằng tỷ số của hai kỳ vọng.

Guard PENGWIN flag CT nếu bất kỳ expected bone nào rỗng ở cả ba members. Review flags trước, rồi dùng candidate cho budget còn lại; guard-only chọn ngẫu nhiên phần còn lại. Nếu flags vượt budget, ghi `guard_budget_infeasible`, risk/sensitivity và CI của policy đó không xác định. Xuất case-macro/bone-macro retained M+S, `1−F1`, missing-volume và complete-omission sensitivity. Complete omission là label evaluation riêng: CT có ít nhất một annotated bone với frozen `missing_volume==1`, tách member-mean/Stage 2; nhãn member-mean bằng một yêu cầu cả ba members mất hoàn toàn bone đó. Không đồng nhất label này với flag prediction-only. Không có positive thì sensitivity không xác định.

Review ở đây là loại CT khỏi automatic acceptance; không giả định người đọc sửa đúng hoàn toàn và không quy CT counts thành thời gian bác sĩ. Guard này không chuyển nguyên trạng sang RibFrac vì lesion-negative subjects hợp lệ.

## Tái lập và chạy tiếp

[Config](../../configs/partition_v3.json), [runner](../../scripts/run_partition_v3.py), [graph](../../src/exchange_ml/partition_graph.py), [CV](../../src/exchange_ml/partition_cv.py), [reporting](../../src/exchange_ml/partition_reporting.py). Manifest bind config/code/software, E1 source manifest/tables, prediction paths/hashes và candidate-membership metadata. Shared E1 lock và exclusive output lock ngăn writers cạnh tranh. Prediction SHA-256 được kiểm tra trước/sau extraction và trước cache reuse. P1/P2/presence/VI phải tái hiện đúng E1 trước khi chấp nhận cache.

Case caches được ghi atomic để resume. Thay code/config/software/source/output root sẽ bị từ chối nếu tái dùng manifest cũ: giữ outputs cũ và chọn output root mới. Run đã complete chỉ verify sources/output hashes rồi thoát, không fit lại. Không đổi các file E1–E6 đang được manifests cũ sử dụng.

**User-typed full command, chưa được agent chạy:**

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash scripts/submit_partition_v3.sh full
```

Launcher submit CPU smoke rồi CPU full với `afterok`. Dùng 4 CPUs, 24 GiB RAM, time limit 4 giờ; partition của cluster tên `gpu` nhưng script không request GPU. Thời gian chờ do Slurm quyết định; time limit không phải ETA. Smoke tái dùng 2 CT thật `004/254` cho extraction/parity; CV/reporting smoke dùng 20 CT giả và 32 bootstrap draws, không có performance evidence trên dữ liệu thật. Full dùng 40 CT thật, 5×3 folds và 10.000 draws. Hai output roots tách riêng:

```text
outputs/smoke_partition_v3/
outputs/partition_v3_development_v1/
outputs/logs/partition-v3-<job_id>.log
```

Outputs chính: `graph_features.csv`, `graph_pairs.csv`, `phantom_checks.csv`, `runtime.csv`, `oof_scores.csv`, `folds.csv`, `tuning.json`, `fold_models.json`, `models.json`, `final_model.json`, `comparison.csv`, `paired_differences.csv`, `risk_coverage.csv`, các bảng `ct_review_*`/`ct_omission_*`, `summary.json`, `partition_manifest.json`. Runtime extraction gồm đọc ba maps, cleanup, graph và E1 component checks; loại chi phí SHA-256, không phải runtime của riêng descriptor G.

110 ca còn lại chỉ được xuất qua `confirmation_candidates.json` như membership inventory, luôn `confirmation_ready=false`. Trước fresh confirmation cần exposure audit, khóa candidate/model/decision criterion và protocol inference riêng. Không tự mở reserved test 50 CT, không submit 330/440 inference mới, không thêm scorer này vào E6 đang chạy.

## Kiểm tra implementation

**Snapshot kiểm tra implementation trước khi tác giả chạy full:** complete cho implementation và smoke. Full sau đó đã hoàn tất như cập nhật đầu trang; kết quả kiểm tra implementation dưới đây được giữ nguyên:

- **182 tests pass** trong 19,7 giây, gồm 22 tests mới về graph/CV/reporting/runner. Có kiểm tra sentinel chống outer-test leakage, JSON reload, CT ties 3/1 bones bằng enumeration, source-specific omission labels, bootstrap không positive, guard infeasibility, interruption/resume và source/output corruption.
- **20/20 phantom assertions pass.** Smoke extraction trên CT `004/254` tạo đúng 4 bones/12 pairs, P1/P2/presence/VI khớp E1. Không fit trên outcomes của hai CT này; phần CV/reporting dùng 20 CT giả/40 bones và 32 draws.
- Smoke toàn pipeline **13,03 giây**, peak RSS **888.500 KiB (~0,85 GiB)**. Đây là smoke, không phải ETA/RAM guarantee cho full cohort. Manifest hoàn tất, độc lập verify **19 output hashes + 2 cache hashes**; chạy lại hoàn tất chỉ verify rồi thoát.
- Preflight xác nhận 40/80 đúng membership, E1 config/code/table/metadata hashes khớp. Raw prediction hashes được kiểm tra cho 6 maps trong smoke; 120 maps full sẽ được kiểm tra khi chạy full. E1–E5 evidence auditor vẫn **PASS 111 output hashes + 142 cache hashes**.
- `bash -n`, local documentation links và whitespace check pass. `sbatch --test-only` chấp nhận CPU script; không submit full job. Scheduler có thể xếp hàng, chưa có runtime đo cho full.

Các command agent thực sự đã dùng cho validation chính:

```bash
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/run_partition_v3.py check
PYTHONPATH=src:. /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -m unittest discover -s tests -v
OMP_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/time -f 'smoke wall_s=%e peak_rss_kb=%M' /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_partition_v3.py smoke
OMP_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_partition_v3.py smoke
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/audit_e1_e5_evidence.py
bash -n scripts/submit_partition_v3.sh scripts/partition_v3.sbatch
sbatch --test-only scripts/partition_v3.sbatch smoke
```

Bước tiếp theo sau full: đọc [chẩn đoán](../research/partition_improvement_2026/assessment_after_v3.md), chốt revision protocol có giới hạn trước một experiment mới. Kết quả hiện tại chưa cho thấy candidate vượt P1; omission guard không được thử thách bởi cohort này. Fresh confirmation vẫn chờ exposure audit và protocol riêng.
