# Thí nghiệm mới: ưu tiên CT cần kiểm tra từ hai loại bất đồng

23/11/2026. Một thí nghiệm mới đã implement, gồm các phép kiểm tra dưới đây. Full chưa chạy; chưa có kết quả cải thiện để đưa vào paper. Không train thêm model hoặc sửa E6 đang chạy.

## Câu hỏi

Bất đồng về cách chia mảnh xương có ích cho merge/split, còn bất đồng về vùng xương xuất hiện có thể bổ sung tín hiệu bỏ sót. Khi bác sĩ chỉ có thể kiểm tra một phần CT, kết hợp hai loại tín hiệu này có ưu tiên được các ca có nhiều xương bị bỏ sót hơn không?

Hai scores được đưa về thang percentile riêng cho từng loại xương, dùng prediction scores của validation; không dùng lỗi GT để chọn trọng số. Điểm một xương là max của hai percentiles; điểm một CT là max các xương. Mọi policy có guard đều dùng cùng quy tắc ưu tiên xương kỳ vọng nhưng cả ba models không dự đoán được.

| Phép kiểm tra | So sánh thực tế | Diễn giải |
|---|---|---|
| Hai kênh có bổ sung giá trị không? | Kết hợp hai kênh so với chỉ bất đồng cách chia mảnh, cùng omission guard | Primary follow-up: retained missing-volume risk tại ngân sách20%CT. Thấp hơn nghĩa là các ca bỏ sót nặng được đưa vào nhóm review tốt hơn, không giả định review sẽ sửa được lỗi. |
| Có phải chỉ nhờ đổi thang điểm? | Controls chỉ structural, chỉ foreground, bản percentile từng kênh và bản kết hợp; báo10policies | Tách scale alignment khỏi việc bổ sung thông tin. Mỗi policy có CI; paired contrasts của candidate với từng control, không chọn một policy thắng rồi đổi primary. |
| Có ích ở ngân sách và loại lỗi khác không? | Review10%/20%CT; co-report missing volume, original/significant merge-split, one-to-one matching, omission sensitivity, từng scan type | Kiểm tra có trade-off không; không gọi một cải thiện riêng endpoint là thắng mọi loại lỗi. |

## Dữ liệu và mức bằng chứng

40validation CT dùng5folds theoCT để fit prediction-only percentile references;100CT held-out đã được xem trước đó chỉ là phân tích hậu nghiệm dùng reference từ toàn bộ40validation. Không dùng110candidate CTs. Paired bootstrap10.000draws theoCT; giữ undefined omission draws/guard-infeasible budgets. Đây không phải xác nhận độc lập của phương pháp mới.

## Chạy bằng Slurm

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash scripts/submit_triage_policy.sh full
```

CPU-only4CPUs,8GiB,2giờ time limit; launcher submit synthetic smoke rồi full với afterok. Không chiếm GPU. Kết quả ở `outputs/triage_policy_development_v1/`; logs `outputs/logs/dual-triage-<job_id>.log`. Config `configs/triage_policy.json`. Time limit không phải ETA.

Chỉ nêu các comparisons đã implement. Chưa có training pilot/backend mới hoặc kết quả full mới. Nếu policy không bổ sung giá trị thì không đưa nó thành contribution chính; paper hiện tại có thể tập trung completed decomposition/held-out evidence và controls liên quan.
