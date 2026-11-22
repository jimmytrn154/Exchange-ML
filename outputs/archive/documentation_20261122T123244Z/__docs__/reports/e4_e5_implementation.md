# E4–E5: sensitivity, qualitative và chi phí

**Status: complete cho full E4–E5 và sửa schema báo cáo, ngày 20/11/2026.** Job Slurm `2816` hoàn tất với exit code `0:0`, từ 13:46:15 đến 14:08:20 ICT (22 phút 05 giây). Đủ 40 validation + 100 held-out cases, 280 bone units và 4.000 fragment/model rows.

Runner [run_e4_e5.py](../../scripts/run_e4_e5.py) đọc E1 validation và E3 held-out đã hoàn tất. Đây là phân tích CPU, không huấn luyện hoặc inference mới. Người dùng đã chạy lệnh full bên dưới. [Config](../../configs/e4_e5.json) giữ thresholds từ `exps.md`; toàn bộ phân tích này là secondary, không thay method hoặc quyết định primary của E3.

## E4 đã triển khai

- **Normalization × annotation scope:** bốn score case-level `{VI/log N, VI/H} × {all, annotated}`, mỗi score là trung bình ba cặp. Dùng nguyên cleanup 26-connectivity, prune 1.000 mm³ của E1/E3. Risk cố định: mỗi model tổng merge+split chia tổng GT instance count trên anatomy được annotate, rồi mean ba member; báo Stage 2 riêng. Không dùng trung bình AURC per-bone để thay phép so sánh case-level. Xuất bốn contrast chỉ thay một yếu tố, với 10.000 paired case-bootstrap draws phân tầng scan type và percentile CI 95%.
- **Scope:** `O` là vùng ít nhất một prediction có ID anatomy ngoài scope; background không thuộc `O`. Xuất tử/mẫu từng cặp của `|D∩O|/|D|`. Khi `|D|=0`, ratio là N/A; mean chỉ dùng cặp có mẫu số dương và ghi số cặp được dùng. Giữ riêng disagreement ngoài GT foreground, không gọi tất cả voxel ngoài GT là anatomy không được annotate.
- **Risk source:** cùng 12 score và bone units, hai cột member-mean/Stage 2, thứ hạng average-rank khi AURC hòa, ΔAURC và paired CIs FA-v2 so với từng baseline từ các CSV E1/E3 đã hash. Đây không phải cross-backend validation.
- **Prevalence:** binary `merge_split>0` và severity quartile với tổng label mass đúng `0,25n`. Review `ceil(0,1n)`/`ceil(0,2n)` bones. Đồng hạng ở cutoff dùng xác suất chọn đều. Prevalence=0 hoặc risk hằng được ghi không đủ thông tin, không gán enrichment có ý nghĩa.
- **Volume blindspots:** mỗi `(case,bone,model)` có F1≥1−1e−12 nhưng missing-volume>0,10 được xuất riêng; báo số model-units, bone units và ca phân biệt. Bảng fragment bao gồm mọi GT ID, best matched prediction/IoU, thể tích GT và coverage bởi **bất kỳ** foreground prediction cùng anatomy. Matching và các risk được đối chiếu với E1/E3 cho từng bone/model; sai khác làm runner dừng.

## E5 đã triển khai

- Percentile là average rank chia số bone units trong từng scan type trên held-out. Giữ nguyên ba category và thresholds trong `exps.md`. Xuất tất cả eligible candidates cùng cờ selected **trước khi vẽ**; chọn theo priority rồi `(case_id,bone)`. Không có candidate thì ghi `no_eligible_case`, không nới cutoff.
- Hình có CT, GT và ba member; chung crop, hai lát axial có discrepancy lớn nhất sau display alignment và ưu tiên cách nhau theo Z. Crop, chỉ số lát, tọa độ tâm physical và mapping màu được lưu CSV. Màu được align one-to-one bằng overlap để không gom nhiều mảnh split thành một màu; số trên hình là ID gốc. Alignment chỉ phục vụ visualization. Hình giữ orientation của array và ghi rõ điều đó, không tự gán nhãn trái/phải lâm sàng.
- Review giữ `floor(0,8n)`/`floor(0,9n)` bones score thấp nhất, dùng expected risk khi hòa. Xuất actual coverage, reviewed bones, expected reviewed CTs, retained risk, Δ so với random và presence. Số CT kỳ vọng dùng xác suất hypergeometric tại nhóm hòa, không giả định các xương trong một CT được chọn độc lập.
- Runtime dùng số đo E3 hiện có, median/IQR theo scan type cho tất cả records và 10 ca đầu mỗi model. Đó là 10 ca đầu theo thứ tự E3, không phải 10 ca mỗi scan type. E4 đo thêm read/cleanup/components/fragment wall time cho chính analysis mới.

**Giới hạn runtime quan sát được:** E3 đã gộp model load vào `inference_seconds`, và gộp scoring với evaluation. Không có số đo riêng historical model-load/read/cleanup/components/baselines. Các ô này là N/A với status `not_recorded_separately_in_E3`. Peak RSS E3 là process high-water mark tích lũy. Không chạy lại timing GPU và không suy ra các số chưa được đo.

## Outputs và provenance

Thư mục full riêng: `outputs/stage3_h1_v2/e4_e5/`. Không ghi vào E1/E2/E3 cũ.

| File | Nội dung |
|---|---|
| `tables/normalization_scope.csv` | Bốn biến thể case-level, AURC và partition share |
| `tables/normalization_scope_comparisons.csv` | Bốn controlled contrasts, ΔAURC/CI |
| `tables/risk_source.csv` | AURC/rank và FA-v2 contrasts theo risk source |
| `tables/enrichment.csv` | Binary/severity enrichment, prevalence và ceiling |
| `tables/f1_volume_blindspots.csv` | Các model/bone blindspots; `split` = dataset split, `split_risk` = numeric split risk; có header kể cả rỗng |
| `tables/blindspot_summary.csv` | Số ca/xương/model-records phân biệt |
| `tables/fragment_coverage.csv` | Mọi GT fragment × bốn model |
| `tables/scope_case_scores.csv`, `scope_pair_components.csv`, `outside_gt_disagreement.csv` | Dữ liệu audit cho E4a |
| `tables/review_budget.csv` | Bone review budget và số CT kỳ vọng |
| `tables/runtime.csv`, `case_runtime.csv` | Runtime summaries và số đo E4 theo ca |
| `heldout/qualitative_candidates.csv`, `qualitative_selections.csv` | Quy tắc chọn và kết quả chọn |
| `heldout/qualitative_coordinates.csv`, `qualitative_display_mapping.csv` | Tái lập crop/lát cắt/mapping màu |
| `figures/qualitative.pdf`, `qualitative.png` | Hình các ví dụ đạt quy tắc |
| `e4_e5_manifest.json`, `summary.json` | Status, provenance, hashes, counts và limitations |

Các bảng E4 có `split` và `stratum`; validation và held-out luôn riêng. Full có 40+100 ca và 80+200 bone units. File fragment có số hàng phụ thuộc số GT fragments thực tế, không đặt một hằng số giả.

Runner kiểm tra config E1/E3, case membership, code scoring, bảng CSV, inference manifests và GT-cache signatures. Preflight chỉ hash các artifacts nhỏ và kiểm tra size/metadata volume. Khi xử lý từng ca, runner hash lại bốn predictions và GT, đối chiếu nguồn E1/E3, rồi kiểm tra chúng không đổi trong lúc đọc. CT dùng cho hình cũng được hash. Manifest ghi code/config/software, cache và output hashes. Locks ngăn duplicate writers và tránh đọc giữa lúc E1/E3 đang ghi. Resume đòi cùng identity và input hashes; không xóa output khi mismatch. Nếu thay code/config sau một run, giữ output cũ và chọn output root mới.

## Kiểm tra đã chạy — agent-typed

- Toàn bộ **98 tests pass**, gồm 12 tests mới: exhaustive cutoff ties và số CT kỳ vọng; review-risk khớp expected risk curve; matching/coverage khớp evaluator cũ; F1 hoàn hảo vẫn mất thể tích; sai anatomy không được tính coverage; 2×2 không thay risk; constant-risk/N/A; lựa chọn deterministic; renderer và physical coordinates.
- Preflight đạt cho 40 validation + 100 held-out cases.
- Smoke thật dùng validation `004/254`, held-out `001/263`: 4 ca, 8 bones, 84 fragment/model rows, 32 bootstrap draws. Status `complete`; 20 report hashes và 4 cache hashes được kiểm tra. Chạy lại xác nhận cả bốn ca dùng verified cache. Smoke không có ca đạt qualitative cutoffs và ghi đúng `no_eligible_case` cho cả ba category.
- Kiểm tra một ví dụ E5 thật riêng trong `/tmp`: category structural chọn `176/right_hip` theo quy tắc trên CSV held-out. Vẽ hai lát, kiểm tra trực tiếp PNG; đây là representative smoke, không phải full E5.
- Một fixture unit test ban đầu tạo cùng tỷ lệ disagreement ở hai scope nhưng kỳ vọng khác nhau. Đã sửa fixture để tạo thay đổi scope thực sự; không sửa công thức phân tích để ép test pass.
- `bash -n` và `sbatch --test-only` đạt; test-only không submit job full.

Các lệnh kiểm tra (CPU):

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests:scripts MPLCONFIGDIR=/tmp/exchange-e4-e5-test-all-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -m unittest discover -s tests

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 MPLCONFIGDIR=/tmp/exchange-e4-e5-check-mpl \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/run_e4_e5.py --check

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 MPLCONFIGDIR=/tmp/exchange-e4-e5-smoke-mpl \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_e4_e5.py --smoke

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 MPLCONFIGDIR=/tmp/exchange-e4-e5-real-visual-mpl \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/smoke_e5_visual.py

bash -n scripts/e4_e5.sbatch
sbatch --test-only scripts/e4_e5.sbatch full
```

## Full — user-typed, đã thực thi cho job 2816

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mkdir -p outputs/logs
sbatch scripts/e4_e5.sbatch full
```

Job dùng 4 CPU cores, RAM 24 GiB, time limit 4 giờ, partition `gpu` như job evaluation E3; **không request GPU**. E4–E5 không có bước CUDA, nên không cần hai GPU. Nếu bị gián đoạn, submit lại cùng lệnh để resume theo ca. Không chạy hai job vào cùng output root.

Theo dõi bằng `squeue -u "$USER"` và `outputs/logs/e4-e5-<job_id>.log`. Chỉ xác nhận full hoàn tất khi log có `E4/E5 FULL COMPLETE` và manifest có `status=complete`, `smoke_only=false` trong provenance.

**Không cần chạy lại full để sửa schema.** Runner đã thay code nên identity hiện tại khác run 2816; giữ nguyên quy tắc từ chối resume khi code thay đổi. Correction bên dưới là bản sửa báo cáo riêng, không gán code mới cho các caches cũ.

## Sửa lỗi cột split — đã áp dụng

Nguyên nhân: upstream `model_risks.csv` có risk tên `split`; phép ghép `{"split": dataset_split, **row}` làm risk ghi đè dataset membership. Lỗi ở serializer, không thay đổi scores, risk values, bộ chọn qualitative hoặc các summary counts.

- Exporter đổi tên numeric field thành `split_risk` trước khi thêm metadata và từ chối collision còn lại. Empty-table schema cũng có `split_risk`.
- Script [repair_e4_split_column.py](../../scripts/repair_e4_split_column.py) hash-verify các source tables và tái dựng biểu diễn legacy để đối chiếu bản lỗi trước khi thay CSV. Không đọc volume, chạy inference hoặc bootstrap lại.
- Bản sửa có **16 hàng: 9 validation, 7 held-out**; mỗi split có 4 ca và 4 bone units phân biệt. Numeric split risk và mọi giá trị khác được giữ nguyên.
- Chỉ một trong 20 report files gốc đổi bytes. **19 report files khác và 140 caches giữ nguyên**. Audit sau sửa đạt 21 output hashes (20 report files + correction receipt) và 140 cache hashes.
- Nguyên bản manifest, CSV lỗi và runner được giữ byte-for-byte trong `outputs/stage3_h1_v2/e4_e5/report_corrections/split_column_v1/`. `correction.json` ghi parent-manifest hash, original compute signature, source hashes, old/new CSV hashes và correction-code hashes.
- Active manifest cập nhật hash CSV và thêm receipt; giữ nguyên `signature`, execution `provenance`, input hashes, summary và cache hashes. Vì vậy `provenance.code_sha256` vẫn mô tả code thực sự đã tính full run, không phải code exporter sau sửa. Audit code cần xem thêm correction receipt và archived original runner.
- **103 tests pass**. Regression tests có hai dataset splits và numeric risk 0/1; schema rỗng; từ chối collision/hash drift; correction idempotent; khôi phục correction bị dừng sau khi ghi CSV nhưng trước manifest commit.

Agent-typed commands đã chạy cho sửa lỗi (CPU, sửa một bảng nhỏ):

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests:scripts MPLCONFIGDIR=/tmp/exchange-e4-repair-test-all-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -m unittest discover -s tests
OPENBLAS_NUM_THREADS=1 /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/repair_e4_split_column.py --check
OPENBLAS_NUM_THREADS=1 /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/repair_e4_split_column.py
OPENBLAS_NUM_THREADS=1 /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/repair_e4_split_column.py --check
```

Lần check cuối trả `already_corrected`; không sửa tiếp artifacts. **User-typed commands cho bản sửa: None.** Không cần GPU hoặc submit lại Slurm.

## Đề xuất cải thiện E5 và trạng thái triển khai

**Cập nhật 20/11/2026:** người dùng đã duyệt triển khai mục 1–3. Supplementary runner đã qua tests và smoke, full chờ user-run Slurm; xem [báo cáo E5 supplementary](e5_supplement_implementation.md). Nội dung đề xuất ban đầu bên dưới được giữ để truy vết. Mục 4 (policy QC mới) vẫn chưa triển khai.

### Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent, plan.
- Origin Date: 2026-11-20.
- Version Label: E5_followup_plan_v1.
- Verification Status: UNVERIFIED cho các phân tích bổ sung; các số dưới đây được đọc từ outputs full hiện có.
- Scope: đề xuất, không thay config/method/budget đã chốt của E3/E5.

### Bằng chứng hiện tại và câu hỏi cần kiểm tra

AURC gộp toàn bộ risk–coverage curve; fixed-budget E5 chỉ xem coverage 0,8/0,9. Ví dụ, member-mean FA-v2/presence ở coverage 0,5 là 0,377996/0,577369, nhưng ở coverage 0,8 là 0,604101/0,604371. Vì vậy AURC đạt không bảo đảm lợi ích tại ngân sách review cụ thể.

Pooled held-out member-mean retained merge+split risk (thấp hơn tốt hơn; point estimates, chưa có fixed-budget CIs):

| Review bones | FA-v2 | Presence | Partition-v1 | Partition-v2 | Random |
|---|---:|---:|---:|---:|---:|
| 10% | 0,595836 | 0,622369 | 0,566454 | 0,598820 | 0,603882 |
| 20% | 0,604101 | 0,604371 | 0,526288 | 0,588215 | 0,603882 |

Nguồn: `tables/review_budget.csv`, `heldout/risk_coverage.csv`. Với Stage 2 và review 20%, FA-v2 là 0,614552 so với presence 0,579879: không được mô tả lợi ích review là robust qua hai nguồn risk.

Ca `187/left_hip` trong panel cùng sai có cả ba member F1=0, missing-volume=1, disagreement=0, nhưng merge+split risk=0. Endpoint hiện tại không đo lỗi bỏ sót nguyên anatomy; đây không phải dữ liệu để suy ra QC mới sẽ cải thiện merge+split risk.

### Các phương án theo thứ tự ưu tiên

1. **Trình bày đủ các phân tích đã định trước.** Đặt full curve cạnh hai budget points; hiển thị cả 10% và 20%, pelvic/femur/pooled, hai risk sources, partition-v1/v2 và random. Giữ ví dụ `176/right_hip`, `160/sacrum`, `187/left_hip`; cải thiện captions A/B/C, arrows/insets và thêm F1/missing-volume bên cạnh M+S. Không đổi mẫu để chọn hình thuận lợi hơn.
2. **Bổ sung độ bất định ở fixed budget.** Giữ nguyên 12 scores, hai budgets, cleanup và endpoint. Tính paired differences retained risk so với presence/random bằng 10.000 case-bootstrap draws, phân tầng scan type, mang tất cả xương và bản sao case; tính lại floor/cutoff ties trong từng draw. Dùng cùng draws cho mọi policy. Xuất CI 95% theo split/stratum/source. Đây là supplementary uncertainty analysis được thêm sau khi xem kết quả; không đổi budget để tìm CI thuận lợi. Mục tiêu là đo độ chính xác của effect, không yêu cầu CI phải thắng.
3. **Đo thêm lỗi bỏ sót và chi phí thực tế.** Giữ M+S làm endpoint cũ, bổ sung `one_minus_f1` và `missing_volume` ở bảng review như secondary/exploratory endpoints. Nhãn đã có trong E1/E3. Luôn báo expected reviewed CTs cạnh reviewed bones: ở budget 40/200 bones, FA-v2/presence/partition-v1/partition-v2 lần lượt review 34/32/37/33 CTs. Một curve với ngân sách CT bằng nhau là extension riêng, không được gọi bảng hiện tại là so sánh cùng chi phí CT.
4. **Nếu muốn cải thiện policy:** khảo sát QC GT-free cho trường hợp cả ba member dự đoán rỗng ở anatomy được kỳ vọng từ scan type; đặt rule và hiệu chỉnh trên validation. Giữ FA-v2 và partition-only làm controls. Đo omission safety bằng F1/missing-volume và review cost, đồng thời báo M+S riêng. Vì ý tưởng được gợi ra từ held-out này, kết quả trên chính 100 ca hiện tại chỉ là exploratory; một claim xác nhận cần dữ liệu mới chưa dùng để thiết kế policy. Không chuyển primary từ FA-v2 sang partition-v1 sau khi thấy bảng trên.

### Compute, outputs dự kiến và bước tiếp

Các mục 1–3 dùng CSV hiện có và CPU; việc vẽ lại chỉ đọc ba ca đã chọn. Chưa có command triển khai cho các phân tích mới, nên không cung cấp lệnh giả. Khi được triển khai, outputs supplementary phải ở thư mục/version riêng, có manifest và nhãn post-heldout; job CPU sẽ dùng Slurm. Nếu một extension sau này cần inference GPU, dùng hai GPU qua Slurm theo yêu cầu của người dùng.

**Next step đề xuất:** làm mục 1–3 trước để E5 trả lời rõ lợi ích/giới hạn và độ bất định; chỉ triển khai thay đổi policy theo mục 4 khi có thiết kế validation và tập xác nhận mới. Chưa chạy bất kỳ phân tích bổ sung hoặc policy mới nào trong lần sửa schema này.
