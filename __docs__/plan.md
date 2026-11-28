# Kế hoạch hiện hành

Cập nhật **28/11/2026**. [progress.md](progress.md) là nguồn tổng hợp kết quả; [exps.md](exps.md) giữ protocol và lệnh chạy. Các roadmap cũ đã được thay bằng kế hoạch này; kết quả và protocol của experiments đã hoàn tất vẫn giữ nguyên.

## Câu hỏi và phạm vi

Khi ensemble phân đoạn CT gãy xương 3D bất đồng, phần **presence** (foreground có xuất hiện không) và **partition** (foreground chung được chia thành instance thế nào) liên quan ra sao tới merge/split, bỏ sót và các lỗi thể tích? Chuẩn hóa và cách đánh giá thay đổi kết luận thế nào?

Working title: *Presence or partition? Decomposing ensemble disagreement for structural-failure detection in 3D fracture CT segmentation.* Mục tiêu submission là **EMBC 2027 full contributed paper**; xác minh lại deadline, format và chính sách submission từ nguồn chính thức trước khi nộp.

Phạm vi gồm E1–E5 trên PENGWIN, E6 RibFrac theo protocol riêng đã triển khai, và development v3/v4 được ghi rõ là hậu E3. Active learning, hidden PENGWIN test set, fracture-type classification và clinical intervention trial không thuộc paper hiện tại.

## Contributions và giới hạn phải giữ

| Đóng góp | Bằng chứng | Cách phát biểu |
|---|---|---|
| C1: phân rã presence/partition và chẩn đoán mẫu số | E1 + controlled perturbations/count replication E2 | Cho thấy cơ chế nén VI/log N trong setup này; không claim phát minh VI/Jaccard hoặc image-resolution invariance |
| C2: kiểm chứng failure detection trên held-out | E3, 100 CT; FA-v2 đạt criterion pooled so với presence/entropy | Báo đầy đủ partition-only/Hungarian; không claim FA-v2 tốt nhất hoặc thắng riêng ở femur |
| C3: điều kiện đánh giá và blindspots | E4/E5, supplementary fixed-budget và omission | Tách annotation scope, risk source, binary prevalence, severity, matching và chi phí CT; chưa claim review savings |
| Hạ tầng hỗ trợ tái lập | ABBC patch, splits, manifests, hashes, unit checks | Supporting contribution, không thay kết quả khoa học |
| External extension | E6 complete: 240 inference records / 80 CT; audit PASS | Báo paired CIs và 60 positive/20 negative; xem [kết quả](../outputs/ribfrac_fast_v1/README.md), giữ mọi kết quả âm tính |

V3/v4 chưa cải thiện đáng tin cậy so với P1. Không hạ chất lượng P1, đổi baseline hoặc chuyển primary sang hàng thuận lợi sau khi xem kết quả. Không coi các nhánh dùng lại validation hay E3 đã xem là fresh confirmation.

## Việc tiếp theo theo thứ tự

1. **External E6 đã hoàn tất.** Inference 240/240, evaluation 80/80 và audit/cohort PASS; [bằng chứng đã lưu trong repo](../outputs/ribfrac_fast_v1/README.md). Không submit lại training/inference/evaluation.
2. **Diễn giải E6 đã audit.** Dùng đủ 12 scores, paired CIs, negative-case denominators và reconstruction limitations từ bảng complete. Giữ kết quả hỗn hợp, không gộp AURC hai datasets thành một benchmark.
3. **Chốt scientific framing.** Đối chiếu E6 với kết quả E1–E5; giữ primary E3 và giới hạn partition-only. Tác giả đã đồng ý triển khai count ablation và evaluator sensitivity sau rà soát; chạy ở roots riêng theo `exps.md`. Phương pháp hoặc training revision rộng hơn vẫn cần protocol/controls riêng.
4. **Chốt manuscript hiện có.** Bản thảo tiếng Anh/PDF 5 trang đã có trong `paper/`; bổ sung E6 chỉ sau audit, điền author metadata và xác nhận DOI PENGWIN. Giữ mixed/negative findings, backend provenance, mixed-batch E6 amendment và chronology limits. Support files đã gỡ khỏi `paper/` có bản sao ngoài repo theo `paper/README.md`.
5. **Kiểm tra trước submission.** Xác minh venue/format/nguồn tham khảo, claim–evidence, data/model provenance, disclosure, integrity và review; tác giả duyệt bản cuối trước nộp.

Không đặt lịch hoàn thành E6 từ GPU probe. ETA phải dựa trên nhiều epochs thật và lịch chạy member03; inference/evaluation có thể thêm thời gian.

## Quy tắc thực thi và bảo toàn nghiên cứu

- Agent được chạy kiểm tra nhẹ, đọc metadata, audit và unit/smoke tests nhỏ. **Tác giả chạy** full training, preprocessing toàn dataset, conversion, multi-seed campaigns, downloads và profiling dài. Luôn ghi rõ command nào đã chạy, command nào chỉ chuẩn bị.
- Không tự đổi thuật toán, data roles, metric, preprocessing, seeds, protocol hay checkpoint selection để xử lý kết quả bất lợi. Giữ baselines, frozen outputs và provenance. GT chỉ vào evaluator/controlled perturbations, không vào predictor/scorer.
- Phân biệt observed facts, documented facts, assumptions và unresolved checks. Nếu lỗi hoặc thiếu dữ liệu ảnh hưởng thiết kế khoa học, báo rõ trước khi đổi hướng; không thay bằng một nghiên cứu khác.
- GPU production theo Slurm, tôn trọng GPU được cấp, một GPU/task và tối đa hai inference/training tasks đồng thời theo launcher. Không chọn GPU chỉ vì utilization thấp.
- Không xóa checkpoints, manifests, split locks, failed/negative results hoặc provenance như file tạm. Giữ repo code tách khỏi dataset/bulk roots.
- Cập nhật kết quả ở `progress.md`, công việc còn lại ở `plan.md`, protocol/lệnh ở `exps.md`. Không tạo thêm report theo stage trong `__docs__`.

## Hướng sau rà soát và phạm vi đã được triển khai

Nguồn: [rà soát và số liệu chẩn đoán](progress.md#rà-soát-toàn-dự-án-trước-khi-cân-nhắc-chạy-lại--22112026). Sau khi tác giả đồng ý, đã triển khai preflight, count ablation, evaluator sensitivity và báo cáo E6 cohorts; count/evaluator full đã complete ngày23/11; pairedCI chưa chứng minh count vượt P1. Commands Slurm ở [exps.md](exps.md#follow-up-sau-rà-soát-slurm-cpu).

| Ưu tiên | Công việc | Đối chứng / tiêu chí đọc | Chi phí dự kiến |
|---|---|---|---|
| 1 | Khôi phục ABBC submodule mapping/URL và kiểm tra bootstrap; exposure audit 110 candidate CTs | Đúng pin+patch, source/config hashes; ghi rõ case nào từng được xem outcomes | Metadata/CPU, không train |
| 2 | Một ablation P1+absolute predicted count thay G, giữ v4 grouped 5×3 CV/objective/alpha grid | P1 nguyên bản, G, count-only, count-variance, v4; paired CT CIs, anatomy strata; GT complexity chỉ vào hậu kiểm, không predictor | CSV/CPU, không inference mới |
| 3 | Supplementary evaluator sensitivity và kiểm chứng CT policy/omission guard đã có | One-to-one/significant-overlap/missing-volume, giữ primary gốc; cùng CT budgets, guard/no-guard; không gọi validation không có positives là đã xác nhận guard | CPU trên predictions; confirmation có thể cần GPU inference |
| 4 | E6 đã complete/audit; cập nhật diễn giải 60 positive/20 negative external subjects | Giữ 12 controls, fixed scores/checkpoint policy và mọi kết quả âm tính; thêm FP burden/negative CT | Campaign hiện có |
| 5 | Chỉ khi cần training revision: một-seed pilot batch/sampling và stage-time profiling | Cùng warmstart, NPY, patch budget, LR schedule, internal validation; tách batch-size effect và guaranteed-foreground rounding | GPU pilot riêng, tác giả chạy |

**Phương án nhỏ ưu tiên:** ablation P1+count, không mở rộng search sang nhiều heads/features. Điểm pooled count-only validation 0,26328 so P1 0,27059 chỉ là hậu nghiệm không CI, còn femur kém hơn. Chỉ cân nhắc xác nhận sau khi có báo cáo paired differences/strata, kiểm tra có phải complexity proxy, và định trước mức cải thiện có ý nghĩa; không lấy một point estimate làm bằng chứng thắng. Nếu không thêm giá trị, dừng nhánh fusion này và giữ framing phân rã/đánh giá theo bằng chứng hiện có.

**Confirmation gate:** freeze thiết kế/thresholds/policy và audit exposure trước khi inference/đánh giá candidates; ghi chronology mới từ lúc đóng băng. Không tune trên E3 đã xem hoặc official RibFrac external rồi coi lại là untouched confirmation. Endpoint/ROI/threshold/architecture mới phải có output root, config/version và kết quả riêng.

**Training gate:** không restart cả ba seeds chỉ vì còn VRAM. Batch6 hiện tại khác số optimizer updates và guaranteed-foreground slots so batch2; pilot phải kiểm tra chất lượng full-volume trên internal validation, không dùng external để chọn. Không hứa GPU utilization hoặc tốc độ tăng cụ thể trước profiling.

**Phạm vi implementation hiện có:** một pipeline Slurm CPU kiểm tra nguồn rồi chạy count/evaluator ở output riêng; E6 cohort report được submit phụ thuộc E6 evaluation khi cần. Đã có count/evaluator full results; không có training pilot mới, không tiêu thụ 110 candidates, không đổi current E6 backend/schedule hay endpoints E1–E5. Chỉ quyết định pilot/backend revision sau khi các diagnostics cho thấy cần thiết.


## Phạm vi pipeline EMBC được tác giả yêu cầu ngày23/11

1. Đọc ít nhất6bàiEMBC được xuất bản, fulltext và evidence matrix trước đây ở `paper/review/` (đã gỡ khỏi paper ngày28/11; bản sao theo `paper/README.md`), suy luận bar có giới hạn accepted-only.
2. Thực hiện thêm một policy structural/foreground fixed-rule sau failed fusions: `configs/triage_policy.json`, CPU Slurm command ở `exps.md`; full2987 đã complete và chưa xác nhận benefit, giữ ở development evidence, không đưa thành contribution.
3. Viết complete English IEEE manuscript trong `paper/`, dùng completed PENGWIN evidence; kiểm tra integrity, simulated review/revision và local PDF. Không tự nộp/publish. E6 đã hoàn tất và audit; bản thảo hiện tại chưa được cập nhật các external results mới.

## Sau lần refine manuscript/cleanup ngày23/11

Contributions đã chốt theo completed evidence: phân rã coverage/grouping và kiểm tra normalization; benchmark12scores/100CT; sensitivity theo failure definition. Overview dùng CT/predictions thật và real risk curves, có provenance để tái tạo. E6 đã complete evaluation và hash/count audit; cập nhật external findings vào bản thảo là bước tiếp theo, giữ mọi kết quả bất lợi. Điền author/affiliation và xử lý các vấn đề còn lại từ review trước khi chốt bản nộp.
