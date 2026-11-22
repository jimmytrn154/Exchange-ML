# Storage layout

## Checkout đang dùng

Root: `/mnt/sdb/shared/dang.cpm/Exchange-ML/`, trên `/dev/sdb`.

| Đường dẫn trong repo | Vị trí / vai trò |
|---|---|
| `dataset/` | Symlink `../dataset`; dữ liệu PENGWIN ở `/mnt/sdb/shared/dang.cpm/dataset/PENGWIN26_task1_2/` |
| `outputs/stage3_h1/members/member_0X/nnUNet_results/` | Checkpoints Anatomy/Fragment của ba member |
| `outputs/stage3_h1/members/member_0X/validation_predictions/` | Instance maps validation |
| `outputs/stage3_h1/members/member_0X/validation_probabilities/` | Foreground probabilities validation |
| `outputs/stage3_h1/` | Các bảng và manifests H1 v1 |
| `outputs/stage2_backend/` | Checkpoints, predictions và metrics Stage 2 |
| `outputs/archive/` | Các lượt chạy cũ, tách khỏi đầu vào hiện tại |
| `outputs/local_copy_v1/` | Kế hoạch chuyển dữ liệu và bằng chứng kiểm tra SHA-256 |

## Checkpoints smoke backend cũ

Bốn file `checkpoint_best.pth` / `checkpoint_final.pth` trong
`outputs/smoke_f4/` đã được chuyển ra bản lưu ngày 22/11/2026:
`/mnt/sdb/shared/dang.cpm/.cleanup_backups/Exchange-ML-20261122T122129Z/retained-artifacts/`.
Các đường dẫn cũ là symlink; metadata smoke và manifest copy giữ nguyên.
`manifest.json` trong bản lưu ghi vị trí và SHA-256 của từng checkpoint.
Không xóa bản lưu khi còn cần các symlink này. Việc chuyển cùng filesystem
không giải phóng dung lượng ổ đĩa.

## Copy từ nguồn chuong

Nguồn giữ nguyên ở `/mnt/sdb/shared/chuong/`. Các artifacts được copy về đúng
đường dẫn repo, kiểm tra SHA-256 rồi mới thay symlink bằng file/thư mục thật.
[Manifest](../outputs/local_copy_v1/manifest.json) là trạng thái copy hiện tại;
[plan](../outputs/local_copy_v1/plan.json) liệt kê đầy đủ nguồn và đích.

Nếu còn job pending, chạy từ root repo:

```bash
python outputs/local_copy_v1/copy_outputs.py
```

Helper bỏ qua jobs đã xác minh, giữ lại file đích nếu có xung đột và không sửa
nguồn. [storage_map.tsv](storage_map.tsv) giữ ánh xạ gốc `b/NNN` để phục hồi
cả các checkpoint không nằm trong Git.

Nguồn `chuong/b/` cho phép đọc các đường dẫn artifacts đã biết nhưng có thể
không cho liệt kê thư mục. Raw/preprocessed training data không thuộc lần
copy kết quả này; các symlink tương ứng vẫn trỏ về nguồn.

## Outputs cho kế hoạch mới

Theo [exps.md](exps.md), tạo outputs mới trong `outputs/stage3_h1_v2/` với các
nhánh `validation/`, `heldout/`, `perturbations/`, `tables/`, `figures/`.
Giữ outputs v1 và archived runs để đối chiếu; không ghi đè khi triển khai v2.

Dataset và bulk artifacts vẫn được Git ignore. Bản clone chứa code/metadata
không tự mang theo các checkpoint hay volume local. Quy tắc quota 100 GB của
checkout cũ trên `/home/24chuong.ta/` không áp dụng cho vị trí checkout này;
kiểm tra dung lượng thực tế bằng `df -h .` trước khi tạo outputs lớn.
