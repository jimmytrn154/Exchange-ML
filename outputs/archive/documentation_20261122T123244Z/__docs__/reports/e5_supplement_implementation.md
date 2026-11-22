# E5 supplementary: fixed-budget CI, omission và chi phí review

**Status: full supplementary complete và audited, ngày 20/11/2026.** Job Slurm `2832`, exit code `0:0`, chạy 53 giây (21:46:22–21:47:15 ICT). Đủ 40 validation + 100 held-out cases, 280 bones; 936 points, 1.728 contrasts, 43.680 curve rows; 10.000 bootstrap draws. Audit hiện tại đạt 41 output hashes và hai cache hashes, khớp mọi original M+S budget point; đủ ba qualitative examples với coordinates/mapping gốc. Các dòng smoke phía dưới là lịch sử chuẩn bị. [Kết luận khoa học và CI](e1_e5_synthesis.md).

Phần này triển khai các mục 1–3 trong đề xuất E5 sau khi xem held-out. Toàn bộ outputs mang nhãn `exploratory_post_heldout`. Không đổi primary E3, score, cleanup, budget 10%/20%, ba ca qualitative hoặc historical runtime. Policy QC mới (mục 4) chưa được triển khai.

## Code và đầu vào

- [Runner](../../scripts/run_e5_supplement.py), [Slurm CPU](../../scripts/e5_supplement.sbatch), [config](../../configs/e5_supplement.json).
- [Statistics](../../src/exchange_ml/e5_supplement_statistics.py): fixed-budget point estimates, CIs, paired differences, expected reviewed CTs và full curves.
- [Figures](../../src/exchange_ml/e5_supplement_figures.py): coverage curves, risk/CT-cost panels và qualitative có captions A/B/C, F1/M+S/missing-volume, ID gốc và arrows.
- Dùng E1 validation 40 ca và E3 held-out 100 ca, giữ riêng hai split. Parent là `outputs/stage3_h1_v2/e4_e5/` đã sửa schema, gồm execution manifest gốc và reporting correction receipt.

Preflight kiểm tra 21 output hashes và 140 cache hashes của parent, correction archives, code nguồn đã tính run gốc hoặc code correction được receipt chỉ rõ. Sau đó kiểm tra source tables E1/E3, case membership và các risk ranges. Không cần checkpoint hay inference mới.

## Phương pháp

### Fixed budget và bootstrap

Tính cho **12 scores và analytical random**, trên pooled/pelvic/femur, cả member-mean và Stage 2. Ba endpoint giữ định nghĩa E1/E3: `merge_split`, `one_minus_f1`, `missing_volume`. M+S có thể lớn hơn 1 và không bị clip; hai endpoint còn lại phải nằm trong [0,1].

Với `n` bone units, coverage yêu cầu `c∈{0,8;0,9}`, giữ `k=floor(cn)` xương có score thấp nhất. Retained risk là `Σ_i w_i r_i/k`, trong đó `r_i` là risk của xương `i`, `w_i` là xác suất xương đó được giữ khi phá đồng hạng đều ở cutoff. `k=0` ghi N/A. Random là kỳ vọng chọn ngẫu nhiên đúng `k` xương, nên risk kỳ vọng bằng mean risk cả tập; không nhân hệ số AURC `(1−1/n)` vào endpoint này.

Full dùng 10.000 bootstrap draws theo **case**, phân tầng scan type; pooled giữ nguyên số ca pelvic/femur mỗi draw. Mang theo mọi xương và multiplicity của ca được lấy lại. Cùng draw được dùng cho mọi policy, risk source, endpoint và budget trong từng stratum. Tính lại cutoff/tie weights trong từng draw. Xuất:

- Percentile CI 95% của retained risk cho mỗi policy.
- Percentile CI 95% của paired difference `policy−presence` và `policy−random`, không trừ hai CI riêng.
- `ci_status`: `estimated`, `insufficient_cases` (dưới hai independent cases), hoặc `zero_retained_units`. Không gán CI khi không có dữ liệu đủ để diễn giải.

CI là marginal supplementary uncertainty, chưa hiệu chỉnh familywise cho toàn bộ bảng và không tạo thêm confirmatory primary tests. Full bắt buộc mọi merge+split point estimate, budget floor, delta và expected CT cost khớp bảng E5 cũ trong sai số 1e−12; không khớp thì dừng.

### Chi phí và hình

Budget vẫn tính theo **xương**. Expected reviewed CTs dùng xác suất lấy không hoàn lại tại cutoff tie theo quy tắc E5 cũ; đây là point estimate, không phải thời gian người đọc hay policy cùng ngân sách CT. Random cost cũng tính cho đúng số xương review, không lấy các xương độc lập theo Bernoulli.

CSV giữ đầy đủ 12 scores + random. Hình định lượng chọn sáu đường cố định để dễ đọc: FA-v2, presence, mean entropy, partition-v1/v2, random; không chọn theo score thắng trên kết quả mới. Hình có đủ hai budgets, hai risk sources, ba strata và cả hai split. Các paired delta CIs nằm đầy đủ trong CSV; bars trên hình risk/CT-cost là CI của risk tuyệt đối.

Qualitative giữ ba ca đã chọn: `176/right_hip`, `160/sacrum`, `187/left_hip`. Runner không gọi lại bộ chọn để tìm ví dụ mới: copy nguyên candidates/selections của parent, kiểm tra raw prediction/GT/CT hashes và đòi crop/lát cắt/physical coordinates/display mapping khớp parent. Hình mới có một page mỗi category, captions nằm ngang, metric omission bổ sung và arrow xác định từ vùng display-ID discrepancy. Arrow chỉ là chú thích visualization, không phải clinical adjudication. Các ID/màu chỉ dùng cho display; không quay lại scorer.

## Provenance, locks và resume

Outputs mới: `outputs/stage3_h1_v2/e5_supplement_v1/`; smoke riêng ở `outputs/smoke_e5_supplement_v1/`. Không ghi lên E1/E2/E3/E4/E5 cũ.

Manifest riêng `e5_supplement_manifest.json` ghi config/code/software hashes, parent-manifest hash và original compute signature, upstream tables, case membership, mode smoke/full và role post-heldout. Statistics cache theo split có signature, checksum nội dung và hash trong manifest; khi resume kiểm tra cả hash đã lưu và tính đủ/duy nhất của grid cells. Run hoàn tất được kiểm tra hashes trước khi trả về, không tính/vẽ lại.

Shared locks ngăn đọc giữa lúc E1/E3/parent đang ghi; exclusive supplement lock ngăn duplicate writers. Code/config/source identity đổi thì từ chối reuse output. Giữ bản cũ và dùng output root mới khi thay version. Kết thúc run kiểm tra lại toàn bộ parent hashes.

| Output | Nội dung / số hàng full |
|---|---|
| `tables/fixed_budget.csv` | 936 rows: hai split × ba strata × 13 policies × sáu endpoints × hai budgets; point risk, CI, cost và trạng thái CI |
| `tables/paired_differences.csv` | 1.728 rows: 12 scores so với presence/random, cùng grid |
| `tables/risk_coverage.csv` | 43.680 rows trên hai split; mọi policy/endpoint và coverage points |
| `figures/risk_coverage_<split>_<risk>.{pdf,png}` | Full curve và hai budget points |
| `figures/fixed_budget_<split>_<risk>.{pdf,png}` | Risk/CI và expected CT cost; không phải equal-CT-budget comparison |
| `figures/qualitative.pdf` | Ba category, một page mỗi ví dụ trong full |
| `figures/qualitative_<category>.{pdf,png}` | Hình riêng đủ độ phân giải |
| `figures/qualitative_annotations.csv` | Tọa độ và rule của display arrows |
| `heldout/qualitative_*.csv` | Candidates/selections giữ nguyên; coordinates/mapping được đối chiếu parent |
| `summary.json`, `source_parent_manifest.json` | Counts, giới hạn và snapshot execution/reporting provenance nguồn |

## Kiểm tra đã chạy — agent-typed

- **117 tests pass**, gồm 14 tests supplementary về exhaustive ties, case multiplicity, pairing, batch/seed invariance, omission risk ranges, khớp estimand cũ, renderer không sửa arrays, output path protection và phát hiện duplicate grid cells khi resume.
- Preflight đạt trên parent full đã sửa schema và toàn bộ E1/E3 inputs.
- Smoke thật: validation `004/254`, held-out `001/263`, 8 bone units, 32 draws. Xuất 936 point rows, 1.728 contrast rows và 1.248 curve rows; các strata một ca/zero retained được ghi N/A đúng trạng thái.
- Smoke qualitative chỉ render ví dụ đầu tiên đã chọn (`176/right_hip`), không chọn lại ví dụ trên bốn ca smoke. Hai ví dụ còn lại được ghi rõ chưa render trong smoke. Đã xem trực tiếp PNG qualitative và fixed-budget; crop/lát cắt/mapping khớp parent.
- Smoke manifest `complete`; **37 output hashes và hai cache hashes đạt**. Parent manifest, 21 output hashes và 140 caches giữ nguyên. Full supplementary manifest chưa tồn tại.
- `bash -n` và `sbatch --test-only` chấp nhận script. Test-only không submit job và dự báo start time của nó không được dùng làm ETA.

Các lệnh agent đã chạy:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests:scripts MPLCONFIGDIR=/tmp/exchange-e5-supplement-test-all-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -m unittest discover -s tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 MPLCONFIGDIR=/tmp/exchange-e5-supplement-check-mpl \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/run_e5_supplement.py --check
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 MPLCONFIGDIR=/tmp/exchange-e5-supplement-smoke-mpl \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_e5_supplement.py --smoke
bash -n scripts/e5_supplement.sbatch
sbatch --test-only scripts/e5_supplement.sbatch full
```

## Full — user-typed, đã thực thi cho job 2832

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mkdir -p outputs/logs
sbatch scripts/e5_supplement.sbatch full
```

Job dùng 4 CPU cores, 16 GiB RAM, time limit 2 giờ, partition `gpu` như các job CPU evaluation trước; **không request GPU**. Không chạy inference hoặc model load mới. Thống kê dùng CSV; qualitative chỉ đọc ba ca cố định.

Log: `outputs/logs/e5-supplement-<job_id>.log`. Submit lại cùng command để resume nếu gián đoạn, với cùng code/config/inputs. Không có hai writers vào cùng output root.

**Next step:** tác giả duyệt [Results/Discussion draft](results_discussion_draft.md) và [claim matrix](e1_e5_synthesis.md). Full đã hoàn tất; không cần submit lại để tổng hợp. Các M+S fixed-budget CIs so với presence cắt 0; missing-volume differences nghiêng về presence. Đây là supplementary evidence sau held-out, không xác nhận policy mới tốt hơn.
