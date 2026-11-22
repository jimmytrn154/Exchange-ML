# Cải thiện tín hiệu partition: đề xuất cho Exchange-ML

**Ngày 22/11/2026 — đề xuất nghiên cứu, chưa triển khai hoặc đánh giá phương pháp mới.** Tài liệu tổng hợp [chẩn đoán code và validation](diagnosis.md), [nguồn phương pháp tương tự](literature.md) và [thiết kế kiểm chứng](design.md). Không thay primary, code, config hoặc kết quả E1–E6.

## 1. Điểm nên sửa là representation và cách kết hợp các tín hiệu

Partition không phải thành phần không có ích: partition-v1 có held-out M+S AURC 0,2720, partition-v2 0,2978, trong khi FA-v2 là 0,3311. Vấn đề là một tín hiệu cấu trúc tốt đang bị kết hợp với tín hiệu khác theo công thức cố định, đồng thời representation hiện tại có những giới hạn về severity.

Với một cặp member, đặt `A=|U−C|/|U|` là presence disagreement, `C` là shared foreground, `U` là union, và `Q_H=VI/H` trên `C`. Khi union không rỗng:

`FA-v2 = A + (1−A) Q_H`.

Presence vừa đẩy tổng score lên, vừa giảm trọng số của partition. Điều đó không sai về đại số; nhưng hệ số cộng một không được bảo đảm phù hợp với mục tiêu đếm merge/split. Trên 80 validation bones, correlation mô tả của presence với M+S là 0,032, còn với missing volume là 0,509; partition-v1 với M+S là 0,759. Tuy nhiên presence có rho 0,624 riêng left hip. Do đó không kết luận presence vô dụng ở mọi anatomy hoặc loại nó dựa trên pooled association. Các số này là chẩn đoán development, không phải confirmation mới.

Một vấn đề khác: nếu một shared instance bị chia thành `k>1` phần, `VI/H=1` ở mọi `k`. Nó nhận ra có thay đổi partition nhưng không phân biệt multiplicity trong trường hợp này. Ngoài ra VI tính theo voxel mass, còn risk hiện tại đếm sự kiện theo instance. Chia đôi mảnh lớn và mảnh nhỏ có thể tạo cùng count risk nhưng VI khác nhau rất nhiều. E2 hiện chia mảnh lớn nhất nên chưa trực tiếp kiểm tra size-swap này.

**Đề xuất ưu tiên:** giữ ba nguồn thông tin riêng — cấu trúc partition, presence và cảnh báo omission — rồi kiểm tra một structural descriptor nhạy với số counterpart trước khi cân nhắc một combiner nhỏ.

## 2. Candidate cụ thể: số counterpart hiệu dụng theo từng instance

Thay vì thử thêm một mẫu số, tạo graph overlap giữa các instance của hai predictions. Mỗi instance là một node; overlap giữ cả quan hệ một–nhiều và nhiều–một. Đây là hướng gần với mask correspondence trong [Siddiqui et al., WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Siddiqui_Uncertainty_Estimation_in_Instance_Segmentation_With_Star-Convex_Shapes_WACV_2024_paper.pdf), nhưng adaptation dưới đây chưa có bằng chứng tốt hơn trên fracture CT. Không gọi nó là metric mới hoặc một phương pháp đã được literature chứng minh.

Một descriptor mềm có thể thử:

\[
p(j\mid i)=\frac{n_{ij}}{\sum_j n_{ij}},\qquad
d_i=\frac{1}{\sum_j p(j\mid i)^2},
\]

\[
G(P,Q)=\frac12\left[
\frac1{K_P^C}\sum_i(d_i-1)
+\frac1{K_Q^C}\sum_j(d_j-1)
\right].
\]

`n_ij` là overlap voxel count trên shared foreground; `K_P^C`, `K_Q^C` là số instances có mặt trong shared domain. `d_i` diễn tả instance `i` trải trên bao nhiêu counterpart có trọng lượng đáng kể. Đổi ID không đổi score. Đảo hai member không đổi tổng đối xứng. Các instance được macro-average thay vì để mảnh lớn tự động chi phối mọi đóng góp.

Trong phantom một instance chia đều thành `k` phần, `G=(k−1)/2`, thay vì cùng bằng một ở mọi severity như VI/H. Trong size-swap phantom hai mảnh lớn/nhỏ, chia đôi một mảnh cho cùng macro contribution nếu cấu trúc overlap tương ứng như nhau. Đây là tính chất của các toy cases, **chưa phải kết quả ranking thực tế**.

Cần định nghĩa rõ empty/unmatched cases: nếu `C` rỗng, ghi `G=0` theo convention kèm trạng thái không có shared support; không diễn giải thành “hai model đúng”. Instance chỉ xuất hiện ở một member được thể hiện qua presence/detection channel, không giả vờ graph trên `C` đo được omission của nó. Aggregate cả ba cặp theo quy tắc cố định, không loại cặp bất lợi sau khi xem GT.

Descriptor này không chữa được mọi vấn đề. Với split rất mất cân bằng, counterpart nhỏ có thể vẫn bị giảm đóng góp. Macro weighting có thể làm các component nhiễu nhỏ quan trọng quá mức; một overlap yếu cũng có thể tạo quan hệ graph giả. Nó không biết member nào đúng và không phát hiện lỗi giống nhau ở cả ba member. Nếu biến đổi graph không vượt Hungarian hoặc count variance đã có, không được coi đó là cải tiến mới.

## 3. Kết hợp bằng mô hình nhỏ, đúng endpoint

[Zenk et al., Medical Image Analysis 2025](https://inrepo02.dkfz.de/record/294922/files/1-s2.0-S1361841524003177-main.pdf?version=1) cho thấy learned aggregation có thể hữu ích, nhưng cũng có failure với dữ liệu fitting ít. [Guarino et al., CVPR 2026](https://arxiv.org/html/2603.29941v1) khai thác thông tin spatial/meta-aggregation, đồng thời lưu ý vấn đề ít dữ liệu và nhiều chiều. Với 40 development CT, lựa chọn đầu tiên nên là một combiner ít tham số, không phải RF/GMM hoặc quality CNN lớn.

Structural head ban đầu chỉ cần so sánh các feature sets đã chốt, chẳng hạn `P1`, `P1+A`, và `P1+G+A`; chỉ cân nhắc predicted count/volume khi có đối chứng tương ứng. Target vẫn là mean per-member normalized M+S. Có thể dùng regularized regression của `log1p(risk)` và lấy output để xếp hạng; không gọi output đó là xác suất clinical failure đã calibrated. Dữ liệu GT chỉ dùng làm nhãn fitting/evaluation. `n_GT`, GT volume và min-GT-fragment volume không được làm input triển khai.

**Phép thăm dò rẻ nhất** là combiner hai feature `P1+A`, chạy trên CSV hiện có để kiểm tra giả thuyết fixed sum gây hại. Nhưng nếu mô hình chỉ học bỏ qua A, nó chưa tạo ra đóng góp phương pháp đáng kể. **Hướng nghiên cứu phương pháp mình ưu tiên hơn** là descriptor object-macro ở mục 2, vì nó trực tiếp nhắm hai giới hạn cơ chế đã chỉ ra, rồi mới thử calibration ít tham số nếu cần.

Không fit bốn mô hình anatomy riêng với mỗi nhóm chỉ 20 validation units. Dùng số tham số chung nhỏ; kiểm tra anatomy bằng stratified readouts và nuisance-only controls. Presence có thể hữu ích có điều kiện, nhưng đó là giả thuyết cần kiểm chứng. Việc mô hình chỉ học lại P1 là một kết quả âm tính có ích, không phải bằng chứng cho một joint detector mới.

## 4. Omission cần một nhánh khác

[Bayesian QuickNAT](https://doi.org/10.1016/j.neuroimage.2019.03.042) là tiền lệ structure-wise QC; [RCA](https://doi.org/10.1109/TMI.2017.2665165) minh họa cách bổ sung image/reference evidence để ước lượng chất lượng mà không cần GT của ca mới. Nhưng full RCA cần reference bank và compute theo ca, khó là lựa chọn đầu tiên ở đây.

Hướng rẻ hơn cho PENGWIN là cảnh báo khi scan metadata xác định anatomy phải có nhưng cả ba member đều dự đoán rỗng. Giữ structural ranker để xử lý phần còn lại. Cảnh báo chỉ bao phủ subclass đó; shared nonempty mistakes vẫn có thể im lặng. Không dùng cùng empty rule cho RibFrac: các ca không có fracture là negatives hợp lệ và lesion presence không biết trước khi inference.

Hai nhánh phải dùng **một ngân sách CT chung**. Một CT có nhiều xương bị flag vẫn tính một lần. Nếu số flag vượt ngân sách, ghi safety constraint không khả thi ở mức đó; không tăng review budget hoặc đổi threshold để tạo lợi thế giả. Mục tiêu có thể là coverage omission tốt hơn với structural-risk trade-off được chốt trước, chứ không nhất thiết một score thắng mọi endpoint.

## 5. Kiểm chứng theo ba bước

| Bước | Can thiệp và controls | Tiêu chí diễn giải |
|---|---|---|
| Phantom | Size-swap khi count error cố định; tăng split/merge multiplicity; unequal volumes; boundary jitter; tiny overlaps/spurious objects; ID permutation; count replication. So với raw VI, P1/P2, macro entropy, Hungarian và count variance. | Xác nhận đúng tính chất dự định và giữ cả failure modes. Một curve toy đẹp chưa chứng minh failure detection. |
| Development 40 CT | So vài feature sets/combiner nhỏ, giữ mọi xương của một ca trong cùng fold. Nested grouped selection nếu chọn hyperparameters/thresholds; scaling chỉ trên training folds. | So paired out-of-fold AURC với **P1-only bắt buộc**, không chỉ FA-v2/entropy. Kiểm tra strata và metadata/size-only controls. |
| Fresh confirmation | Freeze một candidate, fitting rule, risk và strongest-control contrast trước khi xử lý dữ liệu mới. Báo P1/P2/FA-v2 và các baseline đã định. | ΔAURC với CI paired case-bootstrap; retained M+S, missing volume, complete-omission sensitivity tại cùng 10%/20% CT budget. Giữ bone/case estimands rõ, không đổi primary cũ. |

Kiểm tra split hiện tại xác nhận **110 candidate pool cases, 55 pelvic +55 femur**, ngoài initial train, validation, reserved test và 100 E3 cases. Chưa thấy chúng trong active score/risk CSVs đã kiểm tra. Nhưng phải audit lịch sử exposure trước khi gọi chúng là fresh confirmation: Stage 0 đã dùng GT strata, và kiểm tra file membership không chứng minh mọi outcome chưa từng được xem. Các 100 ca E3 đã ảnh hưởng tới ý tưởng mới nên không còn là confirmation độc lập cho extension này.

Validation graph features cần CPU đọc lại saved predictions/caches, không train lại segmentation. Fitting vài feature cần CPU; inference 110 ca mới cần GPU nếu chưa có predictions. Giữ E6 đang chạy nguyên vẹn; scorer/policy mới phải có version/output/protocol riêng. Nếu outcome RibFrac đã dùng để thiết kế, kết quả trên cùng cohort đó chỉ là exploratory cho phương pháp mới.

Ba members trên 110 ca tương ứng 330 lượt inference; thêm Stage 2 để giữ risk-source sensitivity là 440. Đây là số lượt, không phải ETA. Giữ 50 ca reserved test ở vai trò cũ; không tự mở chúng để bù cho một confirmation set đã bị exposure.

## 6. Lựa chọn đề nghị

Mình ưu tiên **prototype prediction-only graph descriptor + structural head ít tham số, giữ presence tách riêng và có omission guard đúng semantics**. Bước kế tiếp cụ thể là freeze một bộ phantom và validation protocol nhỏ, smoke-test representation, rồi so với P1 bằng case-grouped validation. Chỉ chuyển sang fresh confirmation nếu candidate có lý do cơ chế và bằng chứng development đủ rõ; không bỏ P1 control hoặc chọn endpoint khác chỉ để tìm chiến thắng.

Đây là đề xuất có thể triển khai bằng artifacts hiện có, không phải lời hứa chắc chắn vượt P1. Kết quả nếu không cải thiện cũng phân biệt được ba khả năng: scalar combination sai mục tiêu, partition descriptor chưa tương ứng count risk, hoặc ensemble đã thiếu diversity mà score không thể tự tạo ra.

**Lệnh full-run chuẩn bị/thực thi:** none. **Thay đổi thực nghiệm:** none. **Đầu ra của công việc này:** diagnosis, primary-source ledger, design và bản đề xuất; chưa có model/score mới được đánh giá.
