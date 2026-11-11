# Dự án 20 — ML Release & Drift Monitor

Mục tiêu: hiểu vì sao model qua được kiểm tra ban đầu vẫn có thể giảm chất lượng
sau phát hành, và cách đưa ra quyết định rollback có bằng chứng.
Code, dữ liệu, policy và dashboard đã có sẵn. Chỉ thay đổi registry cục bộ, không
triển khai hay rollback bất kỳ dịch vụ thật nào.

## 1. Vào thư mục

```powershell
cd "C:\Users\LOQ\Documents\Codex\2026-11-06\cj\outputs\ML-Release-Drift-Monitor"
python -m venv .venv
```

Không cần API key hay cài thư viện.

## 2. Kiểm tra code

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kỳ vọng `Ran 24 tests` và `OK`.

## 3. Huấn luyện model

```powershell
.\.venv\Scripts\python.exe monitor.py train
```

Hai model logistic được huấn luyện thật trên dữ liệu giả lập:

- stable-v1: dùng tín hiệu urgency_score.
- candidate-v2: dùng thêm độ dài words; trong mẫu lịch sử, độ dài tương quan với nhãn.

faulty-v0 là bản cố ý đảo trọng số để kiểm tra gate; không phải model thứ ba được
huấn luyện độc lập. Các feature có cấu trúc được tạo sẵn, không phải hệ thống hiểu
ngôn ngữ hay suy luận độ khẩn cấp từ văn bản thật.

Train có 300 dòng, dev 100 dòng, test 100 dòng. Chọn threshold trên dev, không chọn
bằng test. Dữ liệu nhỏ và giả lập nên không dùng điểm số để tuyên bố chất lượng thực.

## 4. Khởi tạo bản đang dùng

```powershell
.\.venv\Scripts\python.exe monitor.py initialize
```

Kỳ vọng active là `stable-v1`, epoch là `1`.

## 5. Kiểm tra model xấu bị chặn

```powershell
.\.venv\Scripts\python.exe monitor.py promote --candidate faulty-v0 --expected stable-v1 --request bad-v0
$LASTEXITCODE
```

Kỳ vọng `decision: blocked`, mã `1`. Đây là kết quả đúng, vì faulty-v0 cố ý sai.
Model đang dùng vẫn là stable-v1. Mã `2` mới biểu thị lỗi đầu vào, đường dẫn hoặc trạng thái.

## 6. Phát hành model đạt gate

```powershell
.\.venv\Scripts\python.exe monitor.py promote --candidate candidate-v2 --expected stable-v1 --request release-v2
```

Kỳ vọng gate `passed`; state có:

```text
active: candidate-v2
fallback: stable-v1
epoch: 2
```

Trên bộ test giả lập, stable-v1 đạt 0.99 accuracy, candidate-v2 đạt 1.00.
Model v2 vẫn có thể giảm chất lượng khi quan hệ dữ liệu thay đổi sau phát hành.

## 7. Theo dõi từng cửa sổ dữ liệu

Chạy từng lệnh theo thứ tự:

```powershell
.\.venv\Scripts\python.exe monitor.py watch --window w01
.\.venv\Scripts\python.exe monitor.py watch --window w02
.\.venv\Scripts\python.exe monitor.py watch --window w03
.\.venv\Scripts\python.exe monitor.py watch --window w04
```

Kết quả mong đợi:

| Window | Nhãn nhìn thấy | Accuracy bản đang dùng | Decision |
| --- | ---: | ---: | --- |
| w01 | 100 | 1.00 | healthy |
| w02 | 0 | null | investigate |
| w03 | 100 | 0.02 | watch |
| w04 | 100 | 0.07 | rollback_recommended |

Mỗi cửa sổ có 100 event. w02 có nhãn trong file nhưng nhãn chỉ có sau thời điểm
as_of, nên không được dùng để tính accuracy. `null` nghĩa là chưa đủ dữ liệu để tính,
không phải 0 điểm hay model sai hoàn toàn.

Drift nghĩa là phân bố feature thay đổi; chưa tự chứng minh độ chính xác giảm.
Đề xuất rollback cần hai cửa sổ xấu liên tiếp, đủ nhãn và bằng chứng model dự phòng
tốt hơn trên cùng event đã có nhãn. Các ngưỡng và khoảng tin cậy chỉ minh họa policy.

## 8. Xem dashboard trước rollback

```powershell
Start-Process reports/dashboard.html
.\.venv\Scripts\python.exe monitor.py status
```

Dashboard là file HTML cục bộ, không cần server. Sau w04, active vẫn là candidate-v2:
`rollback_recommended` chỉ là đề xuất, chưa đổi model.

Thử chạy lại w04:

```powershell
.\.venv\Scripts\python.exe monitor.py watch --window w04
```

Kỳ vọng `replayed: true`, bad streak vẫn là 2. Chạy lặp không tạo thêm bằng chứng.

## 9. Thực hiện rollback cục bộ

Phải làm bước này trước w05 để theo đúng kịch bản:

```powershell
.\.venv\Scripts\python.exe monitor.py rollback --expected candidate-v2 --target stable-v1 --evidence w04 --request rollback-001
```

Kỳ vọng:

```text
active: stable-v1
fallback: null
epoch: 3
```

Lệnh kiểm tra bản đang dùng, model dự phòng và snapshot mới nhất. Request ID giúp
chạy lại cùng lệnh không áp dụng rollback thêm lần nữa.

## 10. Kiểm tra sau rollback

```powershell
.\.venv\Scripts\python.exe monitor.py watch --window w05
.\.venv\Scripts\python.exe monitor.py status
Start-Process reports/dashboard.html
```

Kỳ vọng w05 dùng stable-v1 và decision `healthy`. Vì w05 là cửa sổ dữ liệu mới và
đã trở về phân bố bình thường, kết quả này không chứng minh riêng tác động nhân quả
của rollback; bằng chứng so sánh fallback/current nằm trên cùng event ở w03/w04.

Đọc báo cáo chi tiết:

```powershell
Get-Content reports/w04.json -Encoding utf8
```

Các trường cần hiểu: label_coverage, active_metrics, fallback_metrics, paired_gain,
drift, consecutive_bad_windows, decision. PSI dùng bin cố định từ train; JSD kiểm
tra phân bố channel. Chúng là điểm mô tả, không phải p-value hay bằng chứng nguyên nhân.

## 11. Khi chạy lại hoặc sửa cấu hình

Snapshot window và phiên bản model là bất biến. Chạy lặp nguyên trạng được replay;
thay as_of hoặc nội dung dưới cùng window ID bị từ chối. Policy/reference/code thay
đổi cũng không được áp vào registry cũ.

Muốn bắt đầu kịch bản riêng từ đầu, dùng DB khác, đặt tùy chọn trước tên lệnh:

```powershell
.\.venv\Scripts\python.exe monitor.py --db private/experiment2.db initialize
```

Các lệnh sau của kịch bản đó cũng phải có `--db private/experiment2.db`. Không cần
xóa database cũ. Model và report vẫn dùng thư mục chung; không chạy nhiều bộ dữ liệu
khác nhau đồng thời trong cùng thư mục dự án.

## 12. Đọc code

1. make_data.py: dữ liệu lịch sử và năm kịch bản sau phát hành.
2. model.py: SGD logistic, scaling train-only và chọn threshold trên dev.
3. drift.py: PSI/JSD, missing bucket và join nhãn theo thời điểm.
4. engine.py: gate phát hành và điều kiện đề xuất rollback.
5. registry.py: trạng thái, epoch, request idempotency và ghi transaction.
6. monitor.py: CLI và dashboard HTML.

README giải thích giới hạn: chưa có live traffic, authentication, scheduler,
late-label backfill hay triển khai hạ tầng thật. Bộ mẫu dùng dữ liệu giả lập hoàn toàn.
