# Kế hoạch hiện hành

Cập nhật **22/11/2026**. [progress.md](progress.md) là nguồn tổng hợp kết quả; [exps.md](exps.md) giữ protocol và lệnh chạy. Các roadmap cũ đã được thay bằng kế hoạch này; kết quả và protocol của experiments đã hoàn tất vẫn giữ nguyên.

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
| External extension | E6 đang training, chưa có external effect | Chỉ thêm kết luận sau complete count/hash audit; giữ cả kết quả âm tính |

V3/v4 chưa cải thiện đáng tin cậy so với P1. Không hạ chất lượng P1, đổi baseline hoặc chuyển primary sang hàng thuận lợi sau khi xem kết quả. Không coi các nhánh dùng lại validation hay E3 đã xem là fresh confirmation.

## Việc tiếp theo theo thứ tự

1. **Hoàn tất E6 đang chạy.** Theo dõi tiến độ/checkpoint/dung lượng, giữ code/config/root/software identity và một writer/member. Chỉ xử lý resume/dependencies sau khi xác định writer cũ đã dừng; không submit lại full chain để kiểm tra trạng thái.
2. **Audit và đọc E6.** Cần đủ ba trained final checkpoints, 80 external cases, 240 inference records, complete manifests và hashes; sau đó báo 12 scores, paired CIs, negative-case denominators và reconstruction limitations. Không gộp AURC hai datasets thành một benchmark.
3. **Chốt scientific framing.** Đối chiếu E6 với kết quả E1–E5; giữ primary E3 và giới hạn partition-only. Nếu cần phương pháp mới, viết protocol/controls/confirmation design riêng trước khi chạy; hiện không có nhánh v5 được phê duyệt.
4. **Hoàn thiện manuscript.** Viết Method/Introduction, tích hợp bảng kết quả và hình hiện có, báo mixed/negative findings, backend provenance, mixed-batch E6 amendment và chronology limits. Bản nháp cũ ở archive chỉ là nguồn tham khảo.
5. **Kiểm tra trước submission.** Xác minh venue/format/nguồn tham khảo, claim–evidence, data/model provenance, disclosure, integrity và review; tác giả duyệt bản cuối trước nộp.

Không đặt lịch hoàn thành E6 từ GPU probe. ETA phải dựa trên nhiều epochs thật và lịch chạy member03; inference/evaluation có thể thêm thời gian.

## Quy tắc thực thi và bảo toàn nghiên cứu

- Agent được chạy kiểm tra nhẹ, đọc metadata, audit và unit/smoke tests nhỏ. **Tác giả chạy** full training, preprocessing toàn dataset, conversion, multi-seed campaigns, downloads và profiling dài. Luôn ghi rõ command nào đã chạy, command nào chỉ chuẩn bị.
- Không tự đổi thuật toán, data roles, metric, preprocessing, seeds, protocol hay checkpoint selection để xử lý kết quả bất lợi. Giữ baselines, frozen outputs và provenance. GT chỉ vào evaluator/controlled perturbations, không vào predictor/scorer.
- Phân biệt observed facts, documented facts, assumptions và unresolved checks. Nếu lỗi hoặc thiếu dữ liệu ảnh hưởng thiết kế khoa học, báo rõ trước khi đổi hướng; không thay bằng một nghiên cứu khác.
- GPU production theo Slurm, tôn trọng GPU được cấp, một GPU/task và tối đa hai inference/training tasks đồng thời theo launcher. Không chọn GPU chỉ vì utilization thấp.
- Không xóa checkpoints, manifests, split locks, failed/negative results hoặc provenance như file tạm. Giữ repo code tách khỏi dataset/bulk roots.
- Cập nhật kết quả ở `progress.md`, công việc còn lại ở `plan.md`, protocol/lệnh ở `exps.md`. Không tạo thêm report theo stage trong `__docs__`.
