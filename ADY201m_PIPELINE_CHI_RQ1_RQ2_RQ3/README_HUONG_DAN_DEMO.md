# ADY201m: SQL + Pandas chỉ RQ1–RQ3 — rubric mục 4 và 5 (V1.1)

## Đồng bộ số liệu với pipeline v1.1 (không imputation), chạy lúc 2026-11-04T07:40:03

- RQ1: PRACTICAL EXAM (phủ >=5 nguồn), median |r| ≈ 0.719285 trên 10 nguồn. PROJECT cao hơn nhưng chỉ 2 nguồn. Phân tích độ nhạy: `RQ1_SENSITIVITY.py` → `KET_QUA_SENSITIVITY_RQ1/` (kết luận không đổi).
- RQ2: **Random Forest** cao nhất theo VALIDATION F2 ≈ 0.856648; hạng 2 LightGBM ≈ 0.847517. Chênh lệch nhỏ, 1 lần chia, 1 random_state. Kết quả phụ thuộc phiên bản thư viện; chỉ dùng số liệu của lần chạy này.
- RQ3 (test): Baseline + F2 tuning F2 ≈ 0.840293, recall ≈ 0.859251; SMOTE + F2 tuning F2 ≈ 0.843255, recall ≈ 0.856264; chênh F2 +0.002962. Không có bằng chứng SMOTE cải thiện F2. Tuning ngưỡng làm recall tăng (baseline 0.8399 → 0.8593) nhưng không làm F2 baseline tăng (0.8409 → 0.8403).
- Dữ liệu: 4.812 tổ hợp `(SOURCE_FILE, CODE, Mã học kỳ)` lặp lại đã có sẵn trong Excel gốc; pipeline giữ nguyên các dòng này. `CODE` chỉ được hiểu là mã sinh viên trong từng file môn học.
- Thay đổi so với bản trước: thư mục kết quả đổi tên `KET_QUA_CHAY_V1_1` (thay `KET_QUA_KIEM_THU_FULL`); bảng SQL RQ2/RQ3 được sinh lại từ CSV mới; kết luận RQ3 trong demo/notebook/SQL không còn viết cứng.

## Mục 4: SQL Server / SSMS

1. Giải nén ZIP. Mở **SQL/01_TAO_DATABASE_ADY201m.sql** trong SSMS -> Execute/F5. Script nạp 124791 hàng từ 18 nguồn cleaned và tạo view. Cảnh báo: script có DROP/CREATE các bảng/view demo trong database ADY201m; không chạy trên database đang lưu dữ liệu quan trọng.
2. Mở **SQL/02_DEMO_SQL_TRUY_VAN_RQ123.sql** -> F5 trong cùng connection. Query RQ1 tính Pearson từ bảng; RQ2 có bảng so sánh 5 mô hình và trả về model đứng đầu; RQ3 có bảng 4 kịch bản và tính chênh lệch F2, recall. Các câu SQL đều có chú thích nguồn dữ liệu, giả định và giới hạn.
3. Metrics RQ2/RQ3 lấy **nguyên từ CSV kết quả pipeline FULL** trong thư mục `KET_QUA_CHAY_V1_1` và được SQL truy vấn, tính kiểm tra F2. **SQL không tự train model hay thực hiện SMOTE**.
4. Chưa thử chạy trực tiếp trong SSMS trong môi trường tạo gói này. Trước demo nên chạy F5 và chụp kết quả truy vấn.

## Mục 5: Pandas / Python notebook

1. Mở CMD tại thư mục đã giải nén, chạy `py -m pip install -r requirements_demo.txt`.
2. Demo nhanh: `py DEMO_PANDAS_RQ123.py` hoặc nháy `CHAY_PANDAS_DEMO.bat`; kết quả tạo trong `KET_QUA_PANDAS_DEMO`.
3. Notebook: `py -m notebook PHAN_TICH_PANDAS_RQ123.ipynb` hoặc mở bằng VS Code. Chọn **Run All** để tính toán RQ1 từ 18 CSV, đọc kết quả huấn luyện RQ2/RQ3, kiểm chứng confusion matrix, vẽ 3 biểu đồ và đọc diễn giải lựa chọn phân tích.
4. Notebook đã kiểm thử Run All thành công. Dữ liệu đầu vào là **18 CSV sạch**, chưa phải dữ liệu thô.

## Phân biệt với pipeline thuật toán

Bộ này chỉ phục vụ **rubric SQL + Analysis with Python**. File huấn luyện thật là `ADY201m_RQ1_RQ2_RQ3_PIPELINE.py` nằm trong ZIP `ADY201m_RQ123_FINAL.zip` đã gửi ở cuộc trò chuyện này; nó chạy 5 mô hình + SMOTE trên combined CSV. Bộ demo này sử dụng **kết quả FULL v1.1** của pipeline đó, không ghép kết quả từ các lần huấn luyện khác nhau.
