# Tài liệu dự án

## Runbooks và đánh giá hiện tại

- [reports/ribfrac_fast_implementation.md](reports/ribfrac_fast_implementation.md): E6 tăng batch và NPY cache, checkpoint handoff, lệnh chạy và giới hạn phục hồi. Đọc artifacts tại output root để xác nhận trạng thái mới nhất.
- [reports/e6_implementation.md](reports/e6_implementation.md): protocol E6 RibFrac gốc; không chạy song song với fast run trên cùng member.
- [reports/partition_v4_implementation.md](reports/partition_v4_implementation.md): ablation P1+G và đánh giá cross-validation.
- [reports/partition_v3_implementation.md](reports/partition_v3_implementation.md): graph features và fusion trên development cohort.
- [research/partition_improvement_2026/assessment_after_v3.md](research/partition_improvement_2026/assessment_after_v3.md): chẩn đoán điểm yếu so với partition-only.

## Kết quả và kế hoạch paper

- [reports/e1_e5_synthesis.md](reports/e1_e5_synthesis.md): kết quả E1–E5, diễn giải E4, claim–evidence–limits và audit hiện tại.
- [reports/results_discussion_draft.md](reports/results_discussion_draft.md): bản nháp tiếng Anh cho Results/Discussion, chưa phải manuscript submission-ready.
- [reports/e1_e5_evidence.json](reports/e1_e5_evidence.json): exact source rows, hashes, filters và các summary dẫn xuất; kiểm tra bằng `scripts/audit_e1_e5_evidence.py`.

- [exps.md](exps.md): bắt đầu ở đây để làm E1–E6.
- [reports/e1_implementation.md](reports/e1_implementation.md): E1 đã triển khai, kiểm tra và lệnh chạy đầy đủ.
- [reports/e2_implementation.md](reports/e2_implementation.md): E2 phantom, GT perturbations, count replication và lệnh chạy.
- [reports/e3_implementation.md](reports/e3_implementation.md): held-out inference, tiêu chí đánh giá và kiểm tra hoàn tất.
- [reports/e4_e5_implementation.md](reports/e4_e5_implementation.md): E4–E5 CPU sensitivity, qualitative, smoke và lệnh Slurm full.
- [reports/e5_supplement_implementation.md](reports/e5_supplement_implementation.md): E5 bổ sung sau held-out, CI fixed-budget, omission endpoints, hình và lệnh Slurm CPU.
- [plan.md](plan.md): phạm vi, đóng góp và tiêu chí kết luận của paper.
- [storage_layout.md](storage_layout.md): đường dẫn dữ liệu, checkpoints và copy artifacts.
- [reports/data_description.md](reports/data_description.md): inventory và label semantics.

## Nền tảng và lịch sử

- [research/phase1_scoping/methodology_blueprint.md](research/phase1_scoping/methodology_blueprint.md): thiết kế nghiên cứu.
- [research/phase3_analysis/synthesis.md](research/phase3_analysis/synthesis.md): tổng hợp tài liệu.
- [research/phase2_investigation/annotated_bibliography.md](research/phase2_investigation/annotated_bibliography.md): nguồn tham khảo.
- [reports/history/stage3/stage3_fix_progress.md](reports/history/stage3/stage3_fix_progress.md): bằng chứng sửa backend.
- [reports/history/](reports/history/): báo cáo stage cũ để tra cứu, không dùng làm runbook hiện tại.
- [rule.md](rule.md): quy tắc thực thi và báo cáo.
