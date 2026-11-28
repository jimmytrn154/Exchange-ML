# Exchange-ML

Nghiên cứu bất đồng foreground và cách chia instance để phát hiện lỗi cấu trúc trong phân đoạn CT gãy xương 3D.

Tài liệu hiện hành chỉ gồm ba file:

- [progress.md](__docs__/progress.md): trạng thái, kết quả tổng hợp Stage 0–3/H1/E1–E6/v3/v4, bảng so sánh và giới hạn khoa học.
- [plan.md](__docs__/plan.md): contributions, công việc còn lại và quy tắc thực thi.
- [exps.md](__docs__/exps.md): protocol, configs, output paths và lệnh chạy/kiểm tra.

E1–E5 đã hoàn tất. FA-v2 đạt criterion pooled held-out so với presence/entropy; partition-only vẫn mạnh hơn, và v3/v4 chưa cho thấy cải thiện đáng tin cậy so với P1. E6 RibFrac fast chưa hoàn tất external evaluation; đọc progress và artifacts trước khi kết luận transfer hoặc submit thêm jobs.

E6 đã train xong cả ba members ở epoch 1000. Để chạy tiếp external inference trên
tối đa 2 GPU đồng thời, nối CPU evaluation và audit/cohort report:

```bash
bash scripts/finish_ribfrac_fast.sh
```

| Vị trí | Nội dung |
|---|---|
| `src/exchange_ml/`, `scripts/` | Metrics, data/backend, training, inference và evaluation |
| `configs/`, `splits/` | Protocol, seeds, splits và audit metadata |
| `outputs/stage3_h1/`, `outputs/stage2_backend/` | Checkpoints và kết quả backend/v1 |
| `outputs/stage3_h1_v2/` | E1–E5, held-out, perturbations và reporting evidence |
| `outputs/partition_v{3,4}_development_v1/` | Development artifacts; không phải fresh confirmation |
| `outputs/archive/` | Bằng chứng và tài liệu lịch sử |
| `outputs/local_copy_v1/` | Source/destination copy manifests và SHA-256 |
| `paper/` | Manuscript/PDF, tài liệu tham khảo, hai bảng, hai hình và `build.sh` |

Dataset được nối qua `dataset -> ../dataset`; bulk E6 roots ở `/mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6-fast-v1`. Giữ dữ liệu/checkpoints/manifests và không thay đổi code/config/root của campaign đang chạy.

Backend dùng [environment freeze](environment/exchange-stage2.freeze.txt), Python 3.10 và nnunetv2 2.5.1. [Backend reference](baselines/README.md) ghi checkout ABBC và patch cần dùng.

Audit evidence không chạy lại experiments:

```bash
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/audit_e1_e5_evidence.py
```
