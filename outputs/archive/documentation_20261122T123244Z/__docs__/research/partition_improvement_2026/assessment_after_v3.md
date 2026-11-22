# Đánh giá sau full partition_v3

**22/11/2026. Status: complete cho chẩn đoán; chưa thay phương pháp hoặc chạy experiment mới.** Full job 2908 đã hoàn tất trên 40 validation CT/80 bones. Tài liệu này phân tích artifacts đã có, không refit model, chạy CV/bootstrap mới, đọc volumes hay thay E1–E6/v3 outputs. [Script tái lập](diagnose_partition_v3.py) và [evidence JSON](v3_diagnosis.json) bind hashes nguồn.

## 1. Kết luận cần điều chỉnh

Candidate `P1+G+A` chưa chứng minh cải thiện. Nhưng không nên quy toàn bộ khoảng cách pooled OOF AURC cho representation hoặc kết luận một final model chắc chắn sẽ kém tương tự trên tập mới. Phép gộp OOF hiện tại trộn scores từ năm models có calibration khác nhau; AURC phụ thuộc thứ tự toàn tập nên nhạy với sự trộn đó.

Giữ nguyên kết quả đã khóa: pooled member-mean M+S AURC P1 **0,270595**, G **0,268951**, ridge-P1 **0,381297**, candidate **0,418209**. Candidate trừ P1 **+0,147614**, CI conditional `[+0,052731; +0,247878]`. Đây vẫn là kết quả của procedure pooled OOF đã chạy; không thay nó bằng một aggregate thuận lợi hơn sau khi xem outcomes.

Chẩn đoán đã verify 19 output hashes, 40 cache hashes và recorded code hashes. Tính lại predictions từ 15 saved models khớp với OOF CSV, maximum error 0 trong run kiểm tra. Recorded training/test CT membership khớp từng fold. Không tìm thấy lỗi áp dụng scaler/coefficients trong artifacts này.

## 2. Ridge-P1: mất ranking giữa folds

Trong mỗi fold `f`, ridge-P1 là `s_f = b_f + w_f (P1−mu_f)/sigma_f`. Cả năm `w_f` đều dương nên biến đổi này giữ nguyên thứ tự P1 trong fold đó. Thực tế có **0/534** inversions trên các cặp P1 không hòa trong cùng fold, và same-fold AURC bằng P1 ở cả năm folds.

Khi ghép folds, có **370/2.275** cặp P1 không hòa bị đảo thứ tự. Predicted log-risk ở `P1=0` thay đổi **0,35771–0,39853** tùy model. Khoảng lệch 0,04081 lớn hơn tín hiệu tuyến tính tại median P1=0,001164, khoảng 0,011–0,020 tùy slope. Trong 1.378 cặp zero-risk/P1-zero đối với positive-risk/P1-positive, ridge-P1 đảo 191 cặp.

Ví dụ fold 0 có mean test log-risk **0,56775**, nhưng mean ridge-P1 prediction **0,42238**. Fold 2 có mean test log-risk thấp hơn, **0,30147**, nhưng mean prediction cao hơn, **0,45700**. Gộp các score này tạo đảo thứ tự giữa model scales.

Đây là vấn đề estimand/evaluation design, không phải bằng chứng P1 mất tín hiệu sau một biến đổi đơn điệu. Tài liệu chính thức [scikit-learn cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) cũng cảnh báo metric trên pooled predictions không nhất thiết tương đương CV scoring, đặc biệt khi metric không phân rã theo từng sample. Repo dùng NumPy ridge riêng; nguồn này hỗ trợ nguyên lý evaluation, không phải mô tả dependency của implementation.

Same-fold AURC chỉ là diagnostic. Nó không thay primary hiện tại, không tương đương một ranking chung trên 80 bones và không cung cấp năm folds độc lập để tính một CI đơn giản: các training sets chồng lấp nhau. Cũng không được fit model toàn bộ 40 CT rồi báo in-sample AURC để thay OOF.

## 3. Candidate còn có vấn đề ngoài pooling

| Outer fold | Raw P1 AURC | Candidate AURC | Diễn giải cùng fold |
|---|---:|---:|---|
| 0 | 0,36955 | 0,57101 | Kém hơn |
| 1 | 0,45326 | 0,58290 | Kém hơn |
| 2 | 0,12576 | 0,16493 | Kém hơn |
| 3 | 0,19710 | 0,20374 | Kém hơn |
| 4 | 0,30121 | 0,29134 | Tốt hơn |

Candidate có 85 within-fold P1-order inversions; P1+A có 77. Inversion với P1 không tự nó là lỗi — một ranker tốt hơn có thể cần đảo P1 — nhưng các same-fold AURC cho thấy những thay đổi hiện tại chưa giúp ổn định. Presence coefficient âm trong cả năm models. Không có lý do ép nó dương để đẹp công thức; coefficient này chỉ được học cho endpoint M+S có điều kiện trên các features còn lại.

Ridge đang tối ưu MSE của log-risk, không trực tiếp tối ưu ranking/AURC. OOF MSE của ridge-P1 **0,10881**, candidate **0,10916**, so với predictor dùng outer-training mean **0,13185**. Model có học được tín hiệu regression nhưng mục tiêu đó không bảo đảm review ranking tốt hơn. Hyperparameter alpha cũng được chọn theo MSE nên nhược điểm này tồn tại ở bước selection.

N=40 CT là nhỏ. P1 có median **0,001164**, max **0,04709** ở 255/femur. Điểm này có ridge-smoother diagonal khoảng **0,321–0,405** trong bốn training folds chứa nó, cho thấy leverage lớn. Đây không phải chứng minh quan hệ nhân quả hoặc lý do xóa ca; loại outlier sau khi biết nó làm baseline mạnh sẽ làm sai phép so sánh.

## 4. G bổ sung ít thông tin; guard chưa được thử thách

P1/G Spearman **0,94452** pooled, **0,95645** pelvic, **0,96990** femur. Cả hai cùng zero trên đúng 27/80 bones. Trong nhóm này 26 bones không có M+S; tất cả 53 P1-positive bones đều có M+S. Đây là mô tả trên development đã được quan sát, không phải sensitivity/specificity xác nhận độc lập.

G trừ P1 có pooled M+S ΔAURC **−0,00164**, CI `[-0,01485; +0,01076]`; chưa chứng minh improvement. Các CT review contrasts cũng chưa cho thấy lợi ích rõ. G xử lý tốt các balanced-split/size-swap toy properties nhưng vẫn giảm trọng số tiny unequal overlaps. Internal split evaluator đếm extra overlapping predicted IDs, kể cả contact rất nhỏ; do đó descriptor và count endpoint vẫn chưa hoàn toàn tương ứng.

Có **0 all-members-empty bone flags** và **0 complete-omission-positive CTs**, ở cả hai risk sources. Với dữ liệu này, guard+candidate bằng candidate; không có evidence sensitivity hoặc benefit của guard. Zero complete omissions không đồng nghĩa không có partial omissions.

## 5. Không cần làm P1 yếu giả tạo: nó đã có giới hạn thật

Ví dụ **292/femur**: P1=G=0, member-mean M+S=0 và `1−F1`=0, nhưng **43,31% GT foreground volume bị thiếu**; Stage 2 thiếu 40,95%. Đây là một illustration về coverage, không phải một subset benchmark được chọn để đánh giá method. G cũng không giải quyết trường hợp này.

Các số existing development missing-volume AURC, thấp hơn tốt hơn:

| Score | Pooled | Femur |
|---|---:|---:|
| Presence | 0,02109 | 0,01693 |
| FA-v2 | 0,02557 | 0,03212 |
| P1 | 0,03764 | 0,08229 |
| G | 0,03738 | 0,08241 |

Đây là các point estimates, không suy ra paired significance từ CI riêng. Chúng cho thấy lý do giữ hai mục tiêu structural/coverage: presence có ích với missing volume dù việc đưa nó vào một scalar M+S chưa có lợi.

Các stress tests hợp lệ là: cùng count error nhưng đổi fragment mass; severity tăng có kiểm soát; boundary jitter/tiny contacts; exact contingency replication; partial/common-mode errors. Giữ cả ca P1 làm tốt và kém, áp dụng cùng can thiệp/cleanup cho mọi method, báo toàn bộ levels đã định trước. E2 đã có replication denominator checks; không gọi đó là robustness study mới trên CT resampling. Exact count replication làm P1 đổi qua `log(rN)` trong khi VI và G không đổi; điều này không bảo đảm P1 ranking sẽ kém hơn trên cohort thật.

Không nên giảm chất lượng input riêng của P1, đổi cutoff chỉ cho P1, chọn ca/seed/endpoint khiến nó thua rồi bỏ các kết quả khác, hoặc đổi risk definition sau khi xem hiệu ứng để gọi candidate thắng. Một endpoint clinically meaningful khác có thể là supplementary study mới, áp dụng mọi method và khóa trước outcomes mới; không thay primary đã chạy.

## 6. Điều chỉnh mình đề nghị, theo thứ tự

1. **Sửa cách thiết kế evaluation cho version kế tiếp.** Giữ pooled OOF v3 như kết quả lịch sử và diagnostic. Trong protocol mới, tách fold-local ranking performance khỏi pooled OOF calibration stability; một test xác nhận cuối cùng phải được score bởi một final model đã khóa, cùng scale. Không lấy within-fold diagnostic làm primary thay thế hồi tố. Không đổi seed để tìm một split thắng.
2. **Chỉ thêm một ablation có lý do: P1+G, bỏ A khỏi structural head.** Thiết kế hiện tại có P1, P1+A, P1+G+A nhưng thiếu P1+G, nên chưa tách rõ incremental value của G khi không có A. Nếu thử, chọn hyperparameters bằng ranking objective trong inner CT-grouped folds và dùng model ít tham số; raw P1 và raw G vẫn là controls bắt buộc. Pairwise ranking là một hướng có tiền lệ [Burges et al., 2005](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/), nhưng không cần neural network và không hứa pairwise loss sẽ tối ưu AURC hoặc thắng P1. Nhiều cặp training không tạo thêm CT độc lập. Đây là giả thuyết cho version mới, chưa implement hoặc fit.
3. **Coverage là analysis riêng.** Nếu phát triển review policy, giữ P1 structural channel và A/coverage channel; chốt trước acceptable M+S cost và missing-volume objective ở cùng CT budget. Không thêm weighted sum/threshold dựa trên bảng nào vừa thắng. Unanimous-empty guard quá hẹp để chứng minh giải quyết partial omission; common-mode misses cần evidence độc lập ngoài disagreement.
4. **Dừng scalar fusion nếu ablation có giới hạn đó vẫn không có lợi ích.** Giữ negative result và cân nhắc trình bày paper theo decomposition/evaluation insight, nơi P1 là một phát hiện tốt. Đây là đề nghị scientific/editorial, chưa thay study objective hoặc primary E3. Không cần buộc công thức phức tạp nhất phải thắng để kết quả có giá trị.

Chưa nên launch 110-case confirmation cho candidate v3 hiện tại. Exposure audit, một candidate/protocol cuối đã khóa và tiêu chí effect/trade-off vẫn là điều kiện trước confirmation. Không mở reserved test 50 CT hoặc thay scorer E6 đang chạy.

## Kiểm tra và command ownership

Agent đã đọc artifacts/code, tính lại saved-model predictions và descriptive inversions/correlations, verify recorded hashes; không chạy training/CV/bootstrap mới. Command chính đã chạy:

```bash
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B __docs__/research/partition_improvement_2026/diagnose_partition_v3.py --output __docs__/research/partition_improvement_2026/v3_diagnosis.json
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/audit_e1_e5_evidence.py
```

**Observed:** diagnosis PASS trên 40 CT/15 saved models; E1–E5 audit PASS 111 output hashes/142 cache hashes. **User-typed full commands mới: None.** Bước tiếp theo được đề xuất là một revision protocol có giới hạn như mục 6; chưa có full experiment mới được chuẩn bị hoặc tự chạy trong lượt phân tích này.
