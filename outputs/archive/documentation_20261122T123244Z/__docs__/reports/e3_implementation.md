# E3 held-out PENGWIN

**Cập nhật 20/11/2026: full E3 complete.** Inference array `2802` và evaluation `2803` hoàn tất với exit code `0:0`; đủ 100 cases, 200 bone units và 400 inference records. Tiêu chí pooled so với presence/mean entropy đạt. Partition-only có AURC thấp hơn FA-v2; xem [tổng hợp khoa học và giới hạn](e1_e5_synthesis.md). Các lệnh và kiểm tra preflight/smoke bên dưới ghi quá trình chuẩn bị ngày 19/11, không phải yêu cầu chạy lại full.

## Dữ liệu và tiêu chí kết luận

- [Danh sách 100 ca](../../configs/e3_heldout_cases.csv) có 50 pelvic và 50 femur. Chọn từ 105 ứng viên mỗi nhóm với NumPy seed `20261117`. Loại 40 ca train của trajectory `20260920`, 40 ca validation và 50 ca test. Split Anatomy/Fragment lưu trên đĩa khớp preparation manifest. Chọn ca bằng scan type và ID.
- [Config E3](../../configs/stage3_h1_v2.json) ghi danh sách, SHA-256, nguồn checkpoint và thông số chấm điểm. E1 cho dự phóng nửa độ rộng CI ở 100 ca là 0,0573 với presence và 0,0572 với mean entropy, so với effect validation lần lượt 0,2124 và 0,3252. Dùng 100 theo plan. Các ước tính này xuất phát từ validation.
- ABBC ở commit `ccc625fd000ec2dfce622b4621bf2d0b4ee12796` với patch F1–F9. Bốn model dùng Anatomy `checkpoint_best.pth` và Fragment `checkpoint_final.pth`. Cả tám checkpoint và các file `dataset.json`/`plans.json` đều được kiểm tra trước inference.
- Ba member cho 300 lượt inference có instance maps và xác suất Stage B. Stage 2 cho 100 lượt inference có instance maps. Mỗi process dùng một GPU. Runner ghi file nén lossless, xác thực geometry, dtype, range và round trip rồi lưu SHA-256. Manifest theo model cho phép chạy tiếp từng ca và ngăn hai process ghi cùng model.
- Evaluator dùng scorer và risk chung với E1. Score tính từ predictions, probabilities và scan type. Sau đó GT được đọc để tính risk của từng model và risk trung bình ensemble. Tất cả 12 score dùng cùng 200 đơn vị xương. Báo pooled, pelvic và femur cùng hai nguồn risk. Bootstrap 10.000 lần theo ca, phân tầng scan type và sử dụng cùng ca resample cho mọi score. AURC dùng trapezoid và kỳ vọng trên các score đồng hạng.
- Tiêu chí chính dùng member-mean merge+split risk. `ΔAURC = AURC(FA-v2) − AURC(baseline)` phải `≤ −0.02` và CI 95% phải nằm dưới 0 đối với cả presence và mean entropy. `summary.json` ghi từng so sánh và kết quả theo Stage 2.

## Lệnh chạy

Dùng Slurm cho cả inference và evaluation. Mỗi phần tử của job array chạy một model trên một GPU do Slurm cấp. Array mặc định chạy tối đa hai model đồng thời. Evaluation chỉ dùng CPU.

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mkdir -p outputs/logs
```

**Smoke GPU do người dùng chạy.** Dùng ca validation `004` (pelvic) và `254` (femur). Mỗi model xử lý cả hai ca. Kết quả ở `outputs/smoke_e3/`.

```bash
smoke=$(sbatch --parsable --time=02:00:00 scripts/e3_infer.sbatch smoke) &&
sbatch --dependency="afterok:${smoke%%;*}" scripts/e3_evaluate.sbatch smoke
```

**Full E3 do người dùng chạy sau khi thấy `E3 SMOKE EVALUATION COMPLETE`.** Evaluation tự bắt đầu khi cả bốn model hoàn tất inference thành công. Dùng `--array=0-3%1` để giới hạn một GPU đồng thời. Để resume một model, submit lại index tương ứng (`0`/`1`/`2`/`3` cho `member_01`/`member_02`/`member_03`/`stage2`).

```bash
full=$(sbatch --parsable --array=0-3%2 scripts/e3_infer.sbatch full) &&
sbatch --dependency="afterok:${full%%;*}" scripts/e3_evaluate.sbatch full
squeue -u "$USER"
```

Log từng model ở `outputs/logs/e3-<array_id>_<task_id>.log`. Log evaluation ở `outputs/logs/e3-eval-<job_id>.log`.

Đọc `outputs/stage3_h1_v2/heldout/e3_manifest.json` để xác nhận `complete`. Thư mục này có `main_table.csv` (E1 và E3), `paired_differences.csv` (ΔAURC và CI), `comparison.csv`, `risk_coverage.{pdf,png}`, bảng risk frequency, enrichment, runtime và dữ liệu theo từng ca/xương. Kỳ vọng 200 score rows, 600 pair rows, 800 model-risk rows và 400 inference records. Runtime có thời gian đọc, inference (gồm cả nạp model trong backend), ghi và peak CUDA allocation. Thời gian CPU gộp scoring với risk evaluation.

## Kiểm tra đã làm

- `git apply --check` trước khi áp patch và `git apply --reverse --check` sau khi áp đều đạt. Preflight `scripts/run_e3.py check` xác minh config, case list, checkpoint, ABBC import và metadata model. Dung lượng trống lúc kiểm tra là 89,5 GiB. Các prediction/probability validation của bốn model chiếm khoảng 12 GiB cho 40 ca. E3 tự nén cả instance maps và probabilities khi lưu.
- `PYTHONPATH=src:.:tests:scripts python -B -m unittest discover -s tests` chạy bằng interpreter ở trên. **86 tests pass**, gồm các test E3 về volume round trip, phát hiện file bị hỏng khi resume, tie cutoff và tiêu chí kết luận chính.
- Chạy thử CPU trên predictions validation của `004` và `254` trong thư mục tạm. Xuất 4 score rows, 12 pair rows và 16 model-risk rows. Giá trị khớp E1 trong sai số `1e-10`. Đây là replay từ predictions đã lưu.
- Chạy nhánh xuất báo cáo với 100 ca có score/risk giả lập, đủ 10.000 bootstrap draws. Kiểm tra 12 hàng main table, 504 comparison rows và 462 paired contrasts.
- `scripts/run_e3.py infer --model member_01 --smoke` dừng trước khi inference với `E3 inference needs exactly one CUDA device visible to this process`. Phiên kiểm tra không thấy CUDA device và không có allocation Slurm của người dùng tại thời điểm kiểm tra.
