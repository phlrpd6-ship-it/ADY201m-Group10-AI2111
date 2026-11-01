# Giải thích từng query của `SQL/02_DEMO_SQL.sql`

Tất cả số liệu bên dưới được **đối chiếu bằng Pandas trên 18 CSV cung cấp**. SQL thực tế vẫn cần được chạy thử trong SSMS. `TARGET = 0` nghĩa là Passed/đạt; `TARGET = 1` nghĩa là Unsuccessful/không đạt, bao gồm `Not Passed` và `Attendance Fail`.

## Phần chuẩn bị: `USE ADY201m; GO`

```sql
USE ADY201m;
GO
```

- `USE ADY201m`: chọn database mà file tạo dữ liệu đã sinh ra.
- `GO`: kết thúc một batch lệnh cho SSMS; **không phải** câu lệnh truy vấn dữ liệu.

## Query 1 — Kiểm tra số dòng và số nguồn dữ liệu

```sql
SELECT COUNT(*) AS total_rows,
       COUNT(DISTINCT source_file) AS source_files,
       COUNT(DISTINCT code) AS distinct_codes
FROM dbo.ews_cleaned;
```

**Giải thích theo thứ tự:**

- `FROM dbo.ews_cleaned`: lấy dữ liệu từ bảng điểm đã làm sạch (được nạp từ 18 CSV).
- `COUNT(*)`: đếm tất cả **bản ghi** trong bảng, không phải số sinh viên duy nhất.
- `COUNT(DISTINCT source_file)`: đếm số file nguồn khác nhau, không đếm số dòng.
- `COUNT(DISTINCT code)`: đếm số giá trị `CODE` duy nhất. Một CODE có thể có nhiều bản ghi ở các kỳ/lớp, nên kết quả nhỏ hơn tổng dòng.
- `AS total_rows`, `AS source_files`, `AS distinct_codes`: đặt tên các cột kết quả để dễ đọc.

**Kết quả đối chiếu:** `total_rows = 124791`, `source_files = 18`, `distinct_codes = 15201`.

**Nói khi demo:** "Query đầu tiên dùng COUNT và DISTINCT để xác nhận dữ liệu gồm 124.791 bản ghi đến từ đúng 18 nguồn, với 15.201 mã CODE khác nhau."

**Pandas tương ứng:** `len(df)`, `df['SOURCE_FILE'].nunique()`, `df['CODE'].nunique()`.

## Query 2 — Thống kê trạng thái sau cleaning và TARGET

```sql
SELECT status_clean, target, COUNT(*) AS n,
       CAST(100.0 * COUNT(*) / (SELECT COUNT(*) FROM dbo.ews_cleaned)
            AS DECIMAL(10,2)) AS percent_of_all
FROM dbo.ews_cleaned
GROUP BY status_clean, target
ORDER BY target, status_clean;
```

**Giải thích:**

- `status_clean`: trạng thái đã chuẩn hóa (`Passed`, `Not Passed`, `Attendance Fail`).
- `target`: nhãn nhị phân để dùng cho bài toán dự đoán (0/1).
- `GROUP BY status_clean, target`: gom những dòng có cùng cặp (trạng thái, nhãn).
- `COUNT(*) AS n`: đếm số bản ghi của **mỗi nhóm**.
- `(SELECT COUNT(*) FROM dbo.ews_cleaned)`: subquery đếm **toàn bộ** bản ghi làm mẫu số; vì vậy phần trăm là **trên tổng 124.791 dòng**, không phải trên số dòng của riêng từng TARGET.
- `100.0 * ...`: chuyển thành tỷ lệ phần trăm.
- `CAST(... AS DECIMAL(10,2))`: hiển thị hai chữ số thập phân.
- `ORDER BY target, status_clean`: sắp theo nhãn 0 trước 1, rồi theo tên trạng thái.

| status_clean | target | n | percent_of_all |
|---|---:|---:|---:|
| Passed | 0 | 92.179 | 73,87% |
| Attendance Fail | 1 | 10.085 | 8,08% |
| Not Passed | 1 | 22.527 | 18,05% |

**Nói khi demo:** "GROUP BY giúp nhìn tỷ lệ từng tình trạng học tập; có 92.179 lượt đạt và tổng cộng 32.612 lượt không đạt, bao gồm không qua môn và không đủ điều kiện chuyên cần."

**Pandas tương ứng:** `df.groupby(['STATUS_CLEAN','TARGET']).size()` và `100*n/len(df)`.

## Query 3 — RQ1: Điểm Practical Exam-30 theo kết quả học tập

```sql
SELECT target,
       COUNT(*) AS students_with_score,
       CAST(AVG(CAST(score AS FLOAT)) AS DECIMAL(10,3)) AS avg_practical_exam_30
FROM dbo.v_rq1_prefinal_long
WHERE assessment_component = N'PRACTICAL EXAM-30'
GROUP BY target
ORDER BY target;
```

**Giải thích:**

- `dbo.v_rq1_prefinal_long`: một **view** chuyển các cột điểm trước thi cuối kỳ thành dạng dài: mỗi dòng tương ứng một bản ghi–thành phần điểm có giá trị, chỉ giữ hàng `has_prefinal = 1`. Vì vậy `score` ở đây là điểm của component và không NULL.
- `WHERE assessment_component = N'PRACTICAL EXAM-30'`: chỉ chọn điểm thực hành trọng số 30; tiền tố `N` khai báo chuỗi Unicode trong SQL Server.
- `GROUP BY target`: chia làm hai nhóm đạt (0) và không đạt (1).
- `COUNT(*) AS students_with_score`: đếm **bản ghi có điểm** của thành phần đó, không nhất thiết là số sinh viên duy nhất.
- `AVG(CAST(score AS FLOAT))`: tính điểm trung bình, ép kiểu FLOAT để phép tính rõ ràng.
- `CAST(... AS DECIMAL(10,3))`: hiển thị trung bình đến ba chữ số thập phân.
- `ORDER BY target`: in kết quả nhóm đạt trước.

| target | students_with_score | avg_practical_exam_30 |
|---|---:|---:|
| 0 | 30.987 | 5,834 |
| 1 | 12.145 | 0,796 |

**Nói khi demo:** "Trong nhóm bản ghi có điểm Practical Exam-30, nhóm đạt có điểm trung bình khoảng 5,834 còn nhóm không đạt khoảng 0,796. Đây là khác biệt điểm trung bình, **chưa phải hệ số tương quan**."

**Pandas tương ứng:** lọc `eligible['PRACTICAL EXAM-30'].notna()` rồi `groupby('TARGET')['PRACTICAL EXAM-30'].agg(['size','mean'])`.

**Phân biệt:** SQL query 3 chỉ là ví dụ mô tả một điểm thành phần. **Phân tích RQ1 đầy đủ** trong Pandas tính tương quan Pearson **riêng từng file**, tổng hợp median |r| theo nhóm; không so sánh trực tiếp `AVG` SQL với hệ số `r` Pandas.

## Query 4 — RQ2: Đếm dữ liệu đủ điều kiện và xem 10 dòng mẫu

Query 4 có **hai câu SELECT** trong cùng một block.

**4a. Đếm số dòng đủ điều kiện:**

```sql
SELECT COUNT(*) AS modeling_ready_rows
FROM dbo.v_ml_prefinal_dataset;
```

- `dbo.v_ml_prefinal_dataset`: view chọn những dòng `has_prefinal = 1`, tức có dữ liệu điểm trước final để làm input mô hình.
- `COUNT(*)`: tổng bản ghi đủ điều kiện; kết quả đối chiếu là **110.704**.

**4b. Xem dữ liệu mẫu (đã cải thiện để có điểm thực hành thay vì toàn NULL):**

```sql
SELECT TOP (10)
       source_file, [PRACTICAL EXAM-30], [ASSIGNMENTS-30],
       [PROGRESS TESTS-30], target
FROM dbo.v_ml_prefinal_dataset
WHERE [PRACTICAL EXAM-30] IS NOT NULL
ORDER BY record_id;
```

- `SELECT TOP (10)`: lấy 10 dòng **mẫu**, không phải chỉ có 10 dòng trong toàn bộ tập dữ liệu.
- `source_file`: kiểm tra dữ liệu đến từ file nào.
- `[PRACTICAL EXAM-30]` và các tên trong `[...]`: tên cột có dấu cách/dấu `-` cần dấu ngoặc vuông trong SQL Server.
- `target`: kết quả thật 0/1, sẽ dùng làm nhãn huấn luyện.
- `WHERE ... IS NOT NULL`: đảm bảo ít nhất cột Practical Exam-30 có điểm trong ví dụ. Các cột điểm khác **vẫn có thể NULL** vì không phải cấu trúc môn học nào cũng giống nhau.
- `ORDER BY record_id`: sắp theo mã bản ghi để kết quả `TOP 10` ổn định. **Đây không phải lấy mẫu ngẫu nhiên.**

**Nói khi demo:** "SQL chuẩn bị 110.704 bản ghi có điểm thành phần. Lệnh TOP 10 chỉ dùng để kiểm tra cấu trúc dữ liệu; việc huấn luyện và so sánh Random Forest với bốn mô hình còn lại nằm ở Python, không nằm trong SQL."

**Pandas tương ứng:** `eligible = df[df['HAS_PREFINAL']]` và `eligible.loc[eligible['PRACTICAL EXAM-30'].notna(), cols].head(10)`. Bản demo Pandas dùng **kết quả huấn luyện đã lưu**, kiểm tra lại F2 từ confusion matrix rồi xếp hạng `sort_values('val_f2')`.

## Query 5 — RQ3: Kiểm tra mất cân bằng lớp

```sql
SELECT target,
       COUNT(*) AS n,
       CAST(100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS DECIMAL(10,2)) AS pct
FROM dbo.v_ml_prefinal_dataset
GROUP BY target
ORDER BY target;
```

**Giải thích:**

- `FROM dbo.v_ml_prefinal_dataset`: chỉ xét dữ liệu **đủ điều kiện làm mô hình**, không dùng 124.791 dòng toàn bộ.
- `GROUP BY target`: gom thành hai lớp 0 và 1.
- `COUNT(*) AS n`: số bản ghi trong mỗi lớp.
- `SUM(COUNT(*)) OVER ()`: lấy **tổng số dòng của tất cả nhóm** bằng window function. `OVER ()` không chia thêm nhóm nào, nên mẫu số ở cả hai dòng đều là 110.704.
- `100.0 * COUNT(*) / ...`: phần trăm của từng nhãn trên tổng modeling-ready.
- `CAST(... AS DECIMAL(10,2))`: làm tròn hiển thị hai chữ số thập phân.
- `ORDER BY target`: hiển thị lớp 0 trước lớp 1.

| target | n | pct |
|---|---:|---:|
| 0 (đạt) | 81.805 | 73,90% |
| 1 (không đạt) | 28.899 | 26,10% |

**Nói khi demo:** "Tập mô hình có tỷ lệ lớp đạt khoảng 73,9% và không đạt 26,1%, tức dữ liệu mất cân bằng. Vì vậy nhóm có thí nghiệm SMOTE và so sánh F2, nhưng SMOTE được áp dụng trong pipeline huấn luyện Python, không có lệnh SQL nào chạy SMOTE."

**Pandas tương ứng:** `eligible.groupby('TARGET').size()` và `100*n/len(eligible)`. Sau đó đọc hai hàng baseline/SMOTE đã lưu, **tính lại** precision, recall và F2 từ confusion matrix để kiểm tra tính nhất quán.

## Các thuật ngữ dễ bị giảng viên hỏi

| Thuật ngữ | Ý nghĩa ngắn |
|---|---|
| `VIEW` | Bảng ảo dùng để truy vấn từ câu SELECT đã định nghĩa, không cần copy dữ liệu thành bảng riêng. |
| `WHERE` | Lọc bản ghi trước khi gom nhóm. |
| `GROUP BY` | Gom các dòng theo giá trị giống nhau để đếm/tính trung bình. |
| `COUNT(DISTINCT ...)` | Đếm các giá trị khác nhau, loại trừ trùng. |
| `AVG` | Trung bình của giá trị không NULL. |
| `CAST` | Chuyển kiểu dữ liệu / kiểu hiển thị số. |
| `TOP (10)` | Chỉ hiển thị 10 hàng đầu theo thứ tự đang chọn. |
| `TARGET` | Nhãn kết quả học tập 0/1, không phải điểm số. |
| `Pearson r` | Đo chiều/mức độ liên hệ tuyến tính giữa điểm và TARGET. Không chứng minh nhân quả. |
| `F2` | Chỉ số cân bằng precision và recall, ưu tiên recall cao hơn. |
| `SMOTE` | Tạo mẫu tổng hợp cho lớp thiểu số trong quá trình huấn luyện, chỉ dùng trên training. |
| `SHAP` | Giải thích ảnh hưởng của đặc trưng đến dự đoán mô hình; hình trong gói là kết quả đã có. |

**Thông điệp kết luận:** SQL dùng để kiểm tra, lọc và tổng hợp dữ liệu; Pandas thực hiện phân tích RQ1 và đối chiếu/biểu diễn kết quả RQ2–RQ3 từ pipeline học máy đã chạy trước.
