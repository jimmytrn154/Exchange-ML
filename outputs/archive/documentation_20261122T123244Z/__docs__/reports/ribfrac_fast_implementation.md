# E6 performance revision: batch 5/6 và NPY cache

**22/11/2026. Status: implementation và smoke hoàn tất; full handoff/train mới chưa chạy.** Tác giả yêu cầu tăng batch 2,5–3 lần và chọn giữ tổng patches gần protocol cũ. Đây là amendment trước khi có external outcomes, có root/provenance riêng; không sửa code/config/checkpoints của E6 đang chạy.

## Kết quả đo thực tế

Plans cũ có batch **2**, patch `[96,160,160]`, 250 train + 50 validation iterations/epoch: 500 train/100 validation patches. Hai processes E6 dùng gần một CPU core liên tục; kernel `/proc` I/O counters cho thấy đọc logical nhiều TB nhưng physical reads nhỏ hơn nhiều, phù hợp với việc lặp lại giải nén từ OS-cached NPZ. GPU utilization thấp ở nhiều snapshot, không phải chỉ thiếu VRAM.

Representative training case `RibFrac1` có float32 shape `[1,333,568,568]`. Ba lượt đọc data+seg NPZ mất **2,24 / 7,78 / 2,00 giây**; mmap cùng patch từ NPY mất **0,0129 giây/lượt**, với values bằng nhau. Đây là một-case I/O smoke, không phải whole-cohort throughput hay epoch speedup. Toàn bộ cache sẽ giữ nguyên dtype/grid/values, không chuẩn hóa hoặc resample lại.

GPU smoke cuối Slurm **2919 completed, exit 0, 1 phút 18 giây**, dùng architecture/patch thật, AMP, loss/deep supervision và forward/backward/optimizer step. Mỗi batch có 2 warmup + 3 measured steps:

| Batch | Compute seconds/step | Patches/second | Peak Torch reserved GiB | Measured headroom GiB |
|---|---:|---:|---:|---:|
| 2 | 0,529 | 3,780 | 6,36 | 15,21 |
| 5 | 1,260 | 3,967 | 19,22 | 2,28 |
| 6 | 1,515 | 3,960 | 19,39 | 2,10 |

Reserved memory không phải toàn bộ `nvidia-smi` usage. Cả 5/6 pass finite loss và headroom tối thiểu 2 GiB trên allocation đo được. Peak không nhất thiết tuyến tính theo batch. Probe ngắn chọn batch 5 theo measured patches/second; khác biệt 5/6 dưới 1%, chưa phải bằng chứng về chênh lệch ổn định. Full launcher benchmark lại trên allocation mới và khóa lựa chọn an toàn 5 hoặc 6; không suy batch tối đa bằng phép nhân VRAM.

**Tăng batch riêng chỉ tăng compute throughput khoảng 5% trong probe cuối.** Cache NPY nhắm bottleneck đọc/giải nén lớn hơn. Chưa có epoch timing sau chuyển nên không hứa ETA mới, không chia ETA cũ cho 2,5 hoặc 3.

Smoke 2916 ban đầu thất bại do harness gọi `train_step` trước khi bật decoder deep supervision, gây tensor/list mismatch. Harness đã sửa để bật đúng training mode như `on_train_start`; 2917 pass. Sau khi chốt lifecycle của mmap để không giữ mở toàn cohort, 2919 xác nhận backend cuối. Không đổi deep-supervision behavior của E6 thật.

Artifacts đo: `outputs/smoke_ribfrac_fast/io_profile.json`, `outputs/smoke_ribfrac_fast/gpu_probe_v2/speed_selection.json`; selection bind helper-code/settings/parent-plan hashes. Probe v1 giữ lại như lịch sử, không dùng để xác nhận hash backend cuối.

## Cấu hình được chuẩn bị

[Config](../../configs/ribfrac_fast.json), [runner](../../scripts/run_ribfrac_fast.py), [backend](../../src/exchange_ml/ribfrac_fast_backend.py), [tests](../../tests/test_ribfrac_fast.py).

- Batch 5: **100 train/20 val iterations**, đúng **500/100 patches/epoch**.
- Batch 6: **83 train/17 val iterations**, **498/102 patches/epoch**, train −0,4%, val +2% do rounding.
- Giữ 1.000 epochs, patch/spacing/architecture, learning-rate schedule theo epoch, three seeds, split và fixed-final-checkpoint policy. Không tự scale LR theo batch. Giảm iterations làm giảm optimizer-update count; cùng số patches không tương đương cùng optimization trajectory.
- Batch Dice và foreground-oversampling rounding thay đổi theo batch; phải ghi trong phương pháp paper. Không gọi kết quả này là nguyên vẹn fixed-batch E6 cũ.
- Chỉ đổi plan batch size, không gọi lại planner để tăng patch size hoặc model capacity. Giá trị 8 GiB trong inherited config là **source planning target**, không phải giới hạn phần cứng 24 GiB mới.
- Dùng `.npy` mmap cache trên NVMe, properties/NPZ aliases trỏ về nguồn cũ. Full cache verify SHA-256 nguồn và values/dtype từng slice trước khi công nhận output. Disk gate dựa trên uncompressed ZIP entry sizes cộng headroom/reserve; không nạp cả cohort vào RAM. Cache 4 workers, train augmentation vẫn một process để giữ checkpoint RNG contract hiện tại.
- Không giữ mở toàn bộ memmaps trong train process: views được nhả theo sample, kernel file cache vẫn tái dùng được. Copy cả `dataset_fingerprint.json` cùng plans/dataset/splits metadata mà nnU-Net cần ở `on_train_start`; không chỉ tạo NPY.
- Bật cuDNN benchmark trong khi giữ deterministic flag; AMP đã có sẵn. Không nâng Torch/CUDA/nnU-Net hoặc thay environment đang dùng. Hardware/workspace changes vẫn có thể ngăn bitwise equivalence.
- Save checkpoint mỗi 5 epochs thay vì 10. Latest resume vẫn bind identity/config/code/software, optimizer/scaler/RNG. Không dùng best checkpoint để chọn external performance.

## Chuyển tiếp có kiểm tra, không ghi đè run cũ

**User-typed full command:**

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash scripts/submit_ribfrac_fast.sh full
```

Chuỗi jobs: **GPU batch probe → CPU NPY cache → CPU handoff → GPU train array → GPU infer array → CPU evaluation**. Mỗi GPU task dùng một GPU được Slurm cấp; training/inference tối đa hai tasks đồng thời. Không chọn GPU dựa trên utilization. Full cache/handoff/training chưa được agent thực thi theo [command ownership](../rule.md).

**Handoff sẽ dừng jobs cũ `2880`, `2881`, `2882`, chỉ sau benchmark/cache pass và checkpoint resume points đọc được.** Kiểm tra owner/command trước cancellation, chờ original writer nhả lock, copy latest/final checkpoint vào root mới và verify hashes. Old source checkpoints giữ nguyên; phần progress chưa được checkpoint có thể cần học lại. Pending original member03 và downstream jobs cũng được cancel để tránh duplicate writers/dependencies cũ.

Weights/optimizer/scaler/RNG và completed epoch được tiếp tục từ snapshot. Warmstart bridge có identity mới, được lưu trong root mới. Members 01/02 có prefix batch2; member03 nếu chưa có checkpoint bắt đầu bằng batch mới. `handoff.json` và `warmstarts/<member>/source.json` ghi chính xác source/signature/hash/epoch của từng member. Không coi old/new versions là các independent seeds bổ sung hoặc trộn kết quả hai roots.

```text
Nguồn giữ nguyên: /mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6
Root tăng tốc:    /mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6-fast-v1
Logs:            outputs/logs/ribfrac-fast-*.log
```

Không đọc external GT trong benchmark, cache hoặc train. Inference/evaluation, instance reconstruction, threshold, risk definitions và 12 score controls giữ nguyên; unit test đối chiếu AST các hàm infer/evaluate/verify với baseline. External 80 cases chỉ được đánh giá sau fixed final epoch như trước.

Sau khi chuyển, đo 5–10 epochs thật để cập nhật ETA và xác định augmentation còn là bottleneck hay không. Nếu timeout, chỉ resubmit khi các fast writers hiện tại đã dừng; giữ checkpoint và cùng code/config/root, xử lý dependencies cũ trước. Không tự resume đồng thời hai writers cho một member.

## Validation đã chạy

- **211 tests pass** (22,28 giây), gồm 10 tests mới. Test NPY bit-preserving/dtype/mmap lifecycle và metadata prerequisites, budget rounding, protected roots, ownership/cache gates trước cancellation, mock Slurm dependencies/failure, bridge source immutability và warm optimizer state. Tiny actual CPU nnU-Net forward/backward batch2 → checkpoint restore → batch5 pass, weights khớp trước update mới, momentum/scaler/RNG được giữ.
- GPU full-patch probe cả 2/5/6 pass sau harness fix; CPU one-case I/O profile pass.
- Shell syntax và Slurm `--test-only` train array pass. Original E6 preprocessing/training **code hashes vẫn khớp**; E1–E5 auditor PASS 111 output/142 cache hashes.
- Full NPY conversion, handoff, resumed production training và fresh inference **chưa chạy**; không lấy smoke compute rate làm epoch ETA.

Agent-typed validation commands chính:

```bash
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/run_ribfrac_fast.py check
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/ribfrac-fast-mpl /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -m unittest discover -s tests -v > /tmp/ribfrac-fast-full-tests.log 2>&1
sbatch --parsable --gres=gpu:1 --time=00:15:00 scripts/ribfrac_fast.sbatch benchmark --output-root /mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/smoke_ribfrac_fast/gpu_probe_v2
sbatch --test-only --array=0-2%2 --gres=gpu:1 --time=7-00:00:00 scripts/ribfrac_fast.sbatch train
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/audit_e1_e5_evidence.py
```

Next step đã được tác giả đồng ý: chạy full command trên để áp dụng performance revision và đo actual epoch throughput. Đây là revision về compute, không thay giả thuyết/metric để tìm kết quả có lợi.
