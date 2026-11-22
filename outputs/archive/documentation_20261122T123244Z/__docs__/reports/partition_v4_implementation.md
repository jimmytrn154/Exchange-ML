# Partition v4: P1+G với pairwise ranking

**22/11/2026 — nhánh development mới được tác giả đồng ý sau khi xem v3.** Mục tiêu là kiểm tra đúng một ablation còn thiếu, `P1+G` không có presence, và tách performance trong từng fold khỏi vấn đề trộn score scales trong pooled OOF. [Chẩn đoán v3](../research/partition_improvement_2026/assessment_after_v3.md) giữ lý do thay đổi. V3 và primary E3 giữ nguyên; v4 không phải kết quả xác nhận độc lập.

## Thuật toán và giả định

Input chỉ có hai prediction-derived features `partition_v1`, `graph_degree`, tái dùng từ full v3 đã hoàn tất. Với training-fold feature mean/std `mu,sigma`, đặt `z_i=(x_i−mu)/sigma`, `s_i=z_i^T beta`. Features constant dùng scale 1. Có đúng hai coefficients không ràng buộc dấu; intercept bằng 0 vì pairwise loss không xác định intercept và không cần dự đoán mức risk tuyệt đối. Score chỉ dùng để ranking, không phải xác suất hay predicted log-risk.

Với các cặp unordered bone units trong training fold, bỏ cặp risk hòa, đặt `t_ij=sign(r_i−r_j)`, trong đó `r` là frozen member-mean normalized M+S. Tối ưu:

`L(beta) = mean_pairs log(1 + exp(−t_ij (z_i−z_j)^T beta)) + alpha/2 * ||beta||²`.

Mỗi cặp không hòa có trọng số bằng nhau. Đây là linear pairwise logistic adaptation có cùng ý tưởng ranking theo pairs như [Burges et al., 2005](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/); không triển khai neural RankNet và không coi nhiều training pairs là nhiều CT độc lập. Pairwise loss không bảo đảm tối ưu AURC; regularization sẽ được chọn trực tiếp theo validation AURC.

Solver deterministic dùng SciPy L-BFGS-B, stable softplus và analytic gradient. Max iterations 1000, tolerance `1e-9`; không âm thầm dùng một model khác nếu optimizer thất bại. Không có cặp risk khác nhau thì nghiệm regularized là zero coefficients, được ghi rõ là không có ranking information. Không sử dụng presence, entropy, metadata, GT counts/volumes hoặc source-specific Stage 2 labels làm features.

## CV và evaluation mới

Mặc định 5 outer folds, 3 inner folds, seed `20261122`, cả CT nằm chung fold và stratified theo scan type. Dùng lại rule phân fold v3 để tránh chọn seed thuận lợi. Mean/std chỉ được fit trên training portion tương ứng. Alpha grid cố định `[0.001,0.01,0.1,1.0]`; alpha là hệ số trên **mean pairwise loss**, không có cùng thang với sum-squared-error ridge của v3. Inner selection lấy arithmetic mean các inner-fold member-mean M+S AURC; ties chọn alpha lớn nhất. Không gộp predictions khác models để chọn alpha.

Final model chọn alpha bằng grouped CV trên toàn bộ development rồi fit 40 CT. Nó được lưu cho reproducibility, không dùng in-sample performance để thay OOF.

Primary v4 là arithmetic mean của outer-fold AURCs, với candidate trừ raw P1 trên pooled member-mean M+S. Cả candidate và controls phải được đánh giá trên cùng từng fold; negative delta có lợi. Giữ 12 scores E1 và raw G làm controls, tổng cộng 14 scores. Xuất riêng pooled/pelvic/femur, member-mean/Stage 2 và M+S/`1−F1`/missing-volume. Mean fold AURC là estimand mới cho v4; không so trực tiếp số tuyệt đối với pooled AURC v3 như thể cùng metric.

Pooled OOF vẫn được xuất như **secondary score-comparability diagnostic**, kèm score distributions và inversion diagnostics. Không coi việc giảm fold-shift artifact là bằng chứng G thêm thông tin hoặc một test model mới tốt hơn P1. Model ranker vẫn có thể khác scale giữa folds; mean-fold ranking evaluation tách phần này ra chứ không chứng minh global calibration.

10.000 paired bootstrap draws resample **CT trong từng outer fold và scan type**, giữ mọi xương/duplicate CT đi cùng. Mỗi draw tính AURC từng fold rồi arithmetic mean, cùng draw cho mọi score/endpoint. CI conditional trên fixed split/models/OOF scores; không refit models, không định lượng toàn bộ training uncertainty. Không dùng SE của năm folds như năm models độc lập vì training sets chồng lấp. Những fold/stratum không đủ units/cases để xác định metric/CI phải có trạng thái undefined, không tự bỏ fold. Giữ nguyên các kết quả xấu hoặc không đủ thông tin.

V4 không thêm CT review policy hoặc claim utility mới. Omission guard v3 có zero flags/positives trên development; các coverage endpoints ở đây là secondary readouts. Một policy structural/coverage dưới cùng CT budget còn cần acceptable M+S trade-off được khóa trước.

## Nguồn dữ liệu và chống thay đổi bằng chứng

[Config](../../configs/partition_v4.json), model `src/exchange_ml/structural_rank.py`, reporting `src/exchange_ml/structural_rank_reporting.py`, runner `scripts/run_partition_v4.py`. Tên modules mới không dùng prefix `partition_` vì v3 provenance dùng glob prefix đó. Không sửa frozen code/config E1–E6/v3.

Runner chỉ đọc CSV/JSON đã có: full v3 manifest/tables/caches và các E1 membership/risk/score references liên quan. Verify provenance self-hash, output/cache/code/config hashes và exact 40 CT/80 bones; alignment P1/G, raw controls và risk labels phải khớp. Không đọc lại raw CT/predictions/GT/probability volumes/checkpoints; không cần GPU hoặc inference mới. V4 không chứng nhận raw input volumes vẫn nguyên byte ở thời điểm mới: nó bind features/labels đã được full v3 xác minh và lưu.

Shared parent lock và exclusive v4 output lock ngăn cạnh tranh writers. Config/code/software/source identity drift sẽ bị từ chối khi tái dùng output root. Atomic CV checkpoint cho phép resume reporting mà không refit; complete rerun verify hashes rồi thoát. Run mới không ghi đè một thư mục có artifacts nhưng thiếu manifest phù hợp.

## Command do tác giả chạy

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash scripts/submit_partition_v4.sh full
```

Launcher submit CPU smoke rồi full với `afterok`; 4 CPUs, 8 GiB RAM, time limit 2 giờ. Cluster partition tên `gpu` nhưng không request GPU. Scheduler quyết định thời gian chờ; time limit không phải ETA. Full experiment chưa được agent chạy.

```text
outputs/smoke_partition_v4/
outputs/partition_v4_development_v1/
outputs/logs/partition-v4-<job_id>.log
```

Smoke audit sources thật nhưng chỉ fit/report trên 20 CT giả/40 bones, giữ 5×3 folds và giảm bootstrap còn 32. Full dùng 40 CT thật/80 bones và 10.000 draws. Synthetic smoke không có performance evidence trên cohort thật.

Outputs dự kiến: `oof_scores.csv`, `folds.csv`, `tuning.json`, `fold_models.json`, `final_model.json`, `comparison.csv`, `paired_differences.csv`, `fold_comparison.csv`, `fold_risk_coverage.csv`, `pooled_comparison.csv`, `pooled_paired_differences.csv`, `score_diagnostics.csv`, CV checkpoint, summary và manifest. Đọc `comparison.csv`/`paired_differences.csv` cho primary v4; pooled tables chỉ là secondary diagnostic.

## Validation và bước tiếp theo

**Status: complete cho implementation/validation; full v4 chưa chạy.** Agent chỉ chạy unit tests, preflight, synthetic smoke, hash audit và Slurm `--test-only`; không tự submit full experiment.

- **201 tests pass** trong 20,51 giây, gồm 19 tests mới. Kiểm tra gradient bằng finite differences, stationarity, extreme margins, risk ties, train-only scaling và outer-test sentinel. Reporter test chứng minh primary/paired CI giữ nguyên dưới positive affine score changes theo fold, còn pooled diagnostic đổi. Bootstrap kiểm tra whole/duplicate CT và paired CI bằng cùng draws tính tay.
- Actual preflight **PASS 40 CT/80 bones**, verify **19 parent outputs + 40 case caches**, E1/v3 source config/code hashes và alignment các raw scores/risk labels. Không đọc volumes.
- Synthetic smoke **PASS 20 CT/40 bones, 5 outer/3 inner folds, 32 bootstrap draws**. Wall time **1,49 giây**, peak RSS **99.816 KiB (~97,5 MiB)**; đây không phải ETA hoặc RAM guarantee cho full. Độc lập verify **14 output hashes** và code hashes; output full v4 chưa tồn tại tại thời điểm kiểm tra.
- Resume tests mô phỏng gián đoạn ở reporting: tái dùng CV checkpoint, không fit lại. Missing recorded checkpoint, rehashed input drift, model/prediction mismatch, source/output corruption và identity drift bị từ chối. Complete rerun chỉ verify và giữ nguyên bytes.
- `bash -n`, local links/whitespace và Slurm `--test-only` pass. E1–E5 auditor tiếp tục PASS **111 output hashes + 142 cache hashes**; v3 artifacts/code không đổi.

Các command agent đã chạy cho validation chính:

```bash
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/run_partition_v4.py check
PYTHONPATH=src:. /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -m unittest discover -s tests -v > /tmp/exchange-partition-v4-tests.log 2>&1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/time -f 'v4_smoke wall_s=%e peak_rss_kb=%M' /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_partition_v4.py smoke
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_partition_v4.py smoke
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/audit_e1_e5_evidence.py
bash -n scripts/submit_partition_v4.sh scripts/partition_v4.sbatch
sbatch --test-only scripts/partition_v4.sbatch smoke
```

Bước tiếp theo được tác giả đồng ý là chạy bounded development này, kiểm tra incremental value so với P1/G và độ ổn định giữa strata. Nếu vẫn không có lợi ích hữu ích, giữ negative result và dừng scalar fusion. Không tự launch 110-case confirmation, mở reserved 50 CT hoặc đổi scorer E6. Fresh confirmation vẫn cần exposure audit và protocol/candidate/decision criterion riêng đã khóa.
