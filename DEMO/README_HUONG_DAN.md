# ADY201m — Demo Pandas và SQL: RQ1, RQ2, RQ3

## 1. File cần dùng

- `demo_pandas.py`: mã Pandas **đã làm lại**, chia rõ RQ1, RQ2, RQ3.
- `CHAY_PANDAS.bat`: chạy Pandas trên Windows.
- `DU_LIEU/cleaned/`: **18 CSV riêng** đã làm sạch; không có bước demo làm sạch, không dùng combined CSV.
- `DU_LIEU/modeling_eligible/feature_info.json`: tên 21 cột điểm trước thi cuối kỳ.
- `DU_LIEU/ADY201m_NEW_RQ_RESULTS_GROUPED/`: kết quả ML đã lưu để làm RQ2/RQ3; có biểu đồ SHAP tham khảo.
- `KET_QUA/`: 7 bảng CSV và 3 biểu đồ PNG **đã chạy thử**, có thể mở trực tiếp để thuyết trình.
- `SQL/01_TAO_DATABASE_ADY201m.sql`: script tạo cơ sở dữ liệu SQL Server từ 18 CSV, có các INSERT sẵn.
- `SQL/02_DEMO_SQL.sql`: chỉ 5 block SQL phục vụ demo (block 4 có 2 SELECT).
- `GIAI_THICH_SQL_TUNG_QUERY.md`: giải thích từng query, kết quả thực tế dự kiến và lời nói mẫu.

## 2. Chạy Pandas

1. Giải nén ZIP, **không tách** `DU_LIEU` khỏi `demo_pandas.py`.
2. Máy cần Python và các thư viện Pandas, NumPy, Matplotlib. Tại thư mục gốc, mở CMD rồi chạy:

   ```cmd
   py -m pip install -r requirements.txt
   py demo_pandas.py
   ```

   Hoặc bấm `CHAY_PANDAS.bat` sau khi cài thư viện.
3. Các bảng/biểu đồ được ghi lại vào `KET_QUA/`.

**Lưu ý:** Pandas dùng `pd.read_csv` đọc từng CSV, sau đó `pd.concat` **trong bộ nhớ** để thống kê xuyên 18 nguồn. Không sinh file combined, không làm sạch lại CSV.

## 3. Pandas đang làm gì theo từng RQ?

### RQ1 — Những thành phần điểm nào có liên hệ với kết quả không đạt?

- Chọn 110.704 dòng `HAS_PREFINAL=True`.
- Duyệt 21 cột điểm trước thi cuối kỳ, phân tích theo **từng `SOURCE_FILE`**.
- Bỏ các cặp có thiếu điểm; chỉ tính Pearson nếu có **ít nhất 30** cặp hợp lệ và cả điểm/TARGET đều có biến thiên.
- `groupby('SOURCE_FILE')`, `Series.corr()`: tương quan giữa điểm thành phần và `TARGET` (0 = Passed, 1 = Unsuccessful).
- `groupby('assessment_family').median()`: tính **median |r|** cho các dạng trọng số của cùng nhóm điểm.
- `RQ1_demo.png`: chỉ hiển thị những nhóm có mặt ở **ít nhất 5 file**, tránh xếp hạng trực quan theo nhóm chỉ xuất hiện trong 1–2 file.

Ví dụ: `PRACTICAL EXAM` median |r| = **0,7193**, xuất hiện trong **10 file nguồn**. Correlation âm cho thấy điểm cao thường đi cùng xác suất không đạt thấp hơn, **không khẳng định quan hệ nhân quả**. `PROJECT` có median |r| cao hơn nhưng chỉ có **2 file**, nên không nên diễn giải là kết luận chung mạnh nhất.

### RQ2 — Trong 5 mô hình, mô hình nào tốt hơn?

- SQL/Pandas chuẩn bị **110.704 dòng** có dữ liệu trước kỳ thi.
- `pd.read_csv('RQ2_model_comparison.csv')` đọc **kết quả các mô hình đã train**: Random Forest, LightGBM, XGBoost, CatBoost, Decision Tree.
- Pandas **tính lại F2** từ `TP`, `FP`, `FN`: `F2 = 5*TP / (5*TP + 4*FN + FP)`; kiểm tra khớp với F2 đã lưu.
- `sort_values('val_f2', ascending=False)`: chọn mô hình theo **VALIDATION F2**, không dựa trên test.

Kết quả: **Random Forest** tốt nhất với validation F2 **0,8540**; test F2 **0,8406** (đã lưu). Không có test F2 của 4 mô hình còn lại trong dữ liệu cung cấp; **không tự suy diễn** giá trị của chúng.

### RQ3 — Mất cân bằng lớp, SMOTE và SHAP có ý nghĩa gì?

- Pandas đếm lớp `TARGET` trong 110.704 dòng: 81.805 đạt (73,90%) và 28.899 không đạt (26,10%).
- `pd.read_csv('RQ3_baseline_vs_smote.csv')` đọc kết quả baseline và SMOTE **đã train**.
- Tính lại precision/recall/F2 từ confusion matrix để kiểm tra, tính `SMOTE F2 - baseline F2`.
- Kết quả test: baseline F2 **0,840797**, SMOTE F2 **0,843255**, chênh **+0,002458** (khoảng **+0,246 điểm phần trăm**). Precision tăng, recall giảm nhẹ.
- Mở `DU_LIEU/ADY201m_NEW_RQ_RESULTS_GROUPED/RQ3_SHAP_summary.png` để tham khảo giải thích mô hình; đây là kết quả SHAP **đã lưu**, không tính SHAP mới.

**Không nói:** "Pandas đã huấn luyện lại 5 mô hình, chạy SMOTE và tạo SHAP". Demo chỉ kiểm tra, xử lý, trực quan hóa các kết quả huấn luyện đã có. SMOTE phải được áp dụng **chỉ trên tập training** trong quy trình huấn luyện, không áp dụng trước khi chia dữ liệu.

## 4. Chạy SQL Server / SSMS

1. Mở `SQL/01_TAO_DATABASE_ADY201m.sql` trong **SQL Server Management Studio** rồi Execute/F5 **một lần**. Lệnh này tạo `ADY201m` nếu chưa tồn tại và **xóa/tạo lại các bảng/view của dự án**, không chạy lên cơ sở dữ liệu đang chứa thông tin cần giữ. Script khoảng 29 MB, có thể tốn thời gian thực thi.
2. Mở `SQL/02_DEMO_SQL.sql`. Có thể chọn từng block rồi nhấn Execute để trình bày lần lượt.
3. Tham khảo giải thích 5 block trong `GIAI_THICH_SQL_TUNG_QUERY.md`.

**Giới hạn kiểm chứng:** Đã chạy Python thành công với toàn bộ 18 CSV và kiểm tra thống nhất các số liệu. **Chưa chạy SQL Server thực tế** trong môi trường này; cần chạy thử file SQL trên máy có SSMS trước khi trình bày.

## 5. Thứ tự demo ngắn gọn

1. SQL block 1–2: tổng số dòng, thống kê trạng thái.
2. SQL block 3: điểm Practical Exam theo kết quả đạt/không đạt.
3. SQL block 4–5: nguồn dữ liệu đưa vào mô hình, kiểm tra mất cân bằng lớp.
4. Pandas: chạy `py demo_pandas.py`; giải thích 3 biểu đồ `RQ1_demo.png`, `RQ2_demo.png`, `RQ3_demo.png`.

**Ngoài phạm vi:** Không có RQ4/RQ5/RQ6, không trình bày code cleaning.
