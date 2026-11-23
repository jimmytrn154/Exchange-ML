# Đối chiếu sáu bài EMBC đã xuất bản với Exchange-ML

**Ngày kiểm tra: 23/11/2026.** Tổng hợp hai source ledgers `embc_recent_fulltext.{md,json}` và `embc_comparison_fulltext.{md,json}`. Đây là mẫu chọn có chủ đích, chỉ gồm bài đã xuất bản. Không có rejected submissions, review scores hoặc denominator toàn hội nghị, nên không suy ra acceptance rate, xác suất nhận bài hay “ngưỡng tối thiểu” chính thức.

## Phạm vi đã đọc và giới hạn phiên bản

Đã đọc methods/results/discussion của sáu bản toàn văn có publication identity EMBC được kiểm chứng. **Năm** bản là conference PDF hoặc author manuscript mang định dạng/acceptance identity tương ứng; bản eczema thứ sáu là preprint đầy đủ, chưa xác nhận chi tiết methods không đổi trong final proceedings. Do đó có năm đối chứng toàn văn mạnh và một đối chứng bổ sung có version gap. McBee2022 và bài radiomics2025 chỉ là context bổ sung, không được cộng vào sáu dòng dưới.

| Bài đã xuất bản | Đóng góp và bằng chứng nhìn thấy | Điều cần giữ đúng khi so với project |
|---|---|---|
| [EffiSegNet,2024](https://doi.org/10.1109/EMBC53108.2024.10782015) | Decoder gọn; Kvasir-SEG1000 images;80/10/10 split; pretrained/scratch ablation; F1 .9552 so .9502 strongest cited comparator. | Một dataset, gain nhỏ, không pairedCI; scores literature không phải toàn bộ cùng-run control. Full author PDF IEEE layout đã đọc. |
| [MI-CES,2025](https://doi.org/10.1109/EMBC58623.2025.11253265) | Selection dựa neurophysiology; ba EEG datasets; hai pipelines; Wilcoxon; hiệu quả thay đổi theo dataset/model/participant. | Không phải segmentation; selected population thay đổi nên không đồng nghĩa benefit dưới cùng review budget. Full accepted author manuscript đã đọc. |
| [Few-Shot Eczema,2024](https://doi.org/10.1109/EMBC53108.2024.10781985) |528 images,428/100 split; SegGPT retrieval/prompt selection; headline mIoU36.69 so U-Net32.60. | Đã đọc full preprint2023, không phải final five-page proceedings. Publication identity/headline được xác nhận; split/details gán cho preprint, không giả định final giữ nguyên. |
| [Artifact Augmentation,2023](https://doi.org/10.1109/EMBC40787.2023.10340997) | Artifact library/blending + YOLO loss; ACROBAT/ANHIR, unseen images/artifact library; test AP50=.65/recall=.61. | Incomplete labels ảnh hưởng FP; ablation khác train/eval conditions; không biến .65−.16 thành causal same-test effect. Full institutional author PDF đã đọc. |
| [Yeast Instance Dataset,2023](https://doi.org/10.1109/EMBC40787.2023.10340268) |493 densely annotated crops;914 cells/971 traps; dataset release + phân biệt IoU/PQ;296/49/148 split. | Không có trained-model performance table; đây là dataset/resource contribution, không phải precedent để một method claim được bỏ strong baselines. Full conference PDF đã đọc. |
| [AI2Seg,2023](https://doi.org/10.1109/EMBC40787.2023.10341074) | Annotation inspection + open GUI;592/2640 crops,5seeds;54 simulated-noise settings,48 vượt AUC>.55; có settings thất bại. | AUC* grid-selected cùng benchmark là optimistic tuning, không độc lập. Một số gain rất nhỏ (.835→.836) không có CI; paper vẫn có tool và error-model contribution. Full conference PDF đã đọc. |

## “Acceptance bar” có thể và không thể kết luận

Mẫu này **không ủng hộ yêu cầu phải thắng mọi baseline, có kiến trúc lớn mới hoặc có external multicenter trial** như một thuộc tính phổ quát của bài EMBC đã xuất bản. Một câu hỏi hẹp nhưng hữu ích, biomedical motivation rõ, intervention/resource/analysis cụ thể, controls có nghĩa và kết luận tương xứng có thể tạo thành một bài hội nghị hoàn chỉnh. Đây là suy luận mô tả từ các đối chứng, không phải công thức bảo đảm accepted.

Ngược lại, việc một số bài không báo CI hoặc có split thiếu rõ ràng không phải lý do để giảm chất lượng kiểm chứng của mình. Không thể lấy gain .005 F1 ở segmentation task khác để kết luận gain .008 AURC với CI cắt0 của project là đủ. Mức novelty cần tương ứng loại claim: bài dataset cần resource; bài tool cần workflow; bài method cần evidence rằng can thiệp thêm giá trị; bài analysis cần câu hỏi và diagnosis có ích ngoài một bug cục bộ.

## Exchange-ML: điểm mạnh và khoảng trống

| Đã có bằng chứng | Chưa được chứng minh |
|---|---|
| Controlled partition/presence mechanisms; exact count-replication checks;40 development CTs;100 held-out CTs;12 fixed controls; paired case-cluster intervals. | FA-v2 tốt nhất trên structural ranking; cải tiến count/fusion vượt P1 đáng tin cậy; femur-specific superiority. |
| Supplementary matching/overlap sensitivity cho thấy thứ hạng phụ thuộc endpoint; qualitative case giải thích common omission; các failed ablations được lưu. | Clinical correction time/savings; omission-safe review; causal utility chỉ từ AURC; transfer khi E6 còn chạy. |
| Có quy trình frozen outputs, manifests và source hashes; primary và post-held-out development tách rõ. | Formal preregistration/timestamped pre-heldout commit chỉ từ hashes;110 candidates tự động được coi là chưa xem outcomes. |

**Framing nên chọn:** một nghiên cứu chẩn đoán và đánh giá *presence versus partition disagreement for structural-failure detection*, trình bày lợi thế đã xác nhận so presence/entropy cùng giới hạn mạnh hơn so partition-only. Việc P1 mạnh hơn FA-v2 và evaluator sensitivity làm giảm chênh lệch là kết quả cần giải thích, không phải giấu. Có thể thêm một QC decision policy hoặc kênh omission mới nếu rõ cơ chế và có output riêng; khi chưa chạy thì manuscript chỉ mô tả ở future work, không tính là contribution thực nghiệm.

**Nhận định thực tế:** cấu trúc bằng chứng hiện có đủ để viết một bản EMBC đầy đủ, nghiêm túc và có phạm vi rõ. Khả năng cạnh tranh phụ thuộc việc reviewer thấy diagnosis mới, liên quan ứng dụng và generalizable đến đâu. Mẫu accepted-only không cho phép nói “EMBC dễ”, “không cần quá thuyết phục”, hoặc khẳng định paper sẽ được nhận. E6 có thể bổ sung phạm vi nếu kết quả có thông tin; không cần bịa hoặc chờ E6 để hoàn thành một draft PENGWIN-only trung thực.

## Quyết định cho bản viết hiện tại

1. Giữ Methods/results của experiments đã hoàn tất và mọi kết quả bất lợi; đóng góp không dựa vào count/fusion gain chưa rõ.
2. Bảng chính gồm strong partition-only controls; scope/endpoint/clinical-utility claims tách riêng.
3. Không đưa acceptance-bar discussion vào technical manuscript như một luận cứ khoa học. Review này chỉ phục vụ planning.
4. Cải tiến mới phải có protocol, controls, output roots và Slurm command riêng; manuscript hiện tại không báo kết quả khi jobs chưa chạy.
5. Dùng template thật,4 trang nội dung và trang thứ5 chỉ references theo EMBC2027 guideline đã xác minh; giữ anonymous local draft nếu tác giả/affiliation chưa được cung cấp. Cần kiểm tra human metadata/disclosure/ethics trước submission.
