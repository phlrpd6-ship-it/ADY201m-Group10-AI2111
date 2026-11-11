# ADY201m - PIPELINE CHỈ RQ1 / RQ2 / RQ3

## Các RQ chính thức

- **RQ1:** Which pre-final assessment components are most strongly associated with an unsuccessful outcome?
- **RQ2:** Which of five algorithms (Decision Tree, Random Forest, XGBoost, LightGBM, and CatBoost) performs best for the early-warning task?
- **RQ3:** Does SMOTE, together with F2-based threshold tuning, improve the recall-oriented performance of the selected model?

Gói này **chỉ chạy 3 câu hỏi trên**. Không chứa bước phân tích hay báo cáo ngoài phạm vi, không cần chạy SHAP.

## Thành phần

- `ADY201m_RQ1_RQ2_RQ3_PIPELINE.py`: mã huấn luyện dựa trên pipeline GROUPED gốc, được lược bỏ phần dư, giữ split theo `(SOURCE_FILE, CODE)`, 21 biến pre-final và 5 thuật toán.
- `DU_LIEU/ADY201m_combined_cleaned.csv`: dữ liệu combined từ gói pipeline gốc; không viết lại, không sửa giá trị.
- `DU_LIEU/18_CSV_GOC/`: 18 file CSV đã làm sạch từ gói pipeline gốc để truy nguồn dữ liệu.
- `KET_QUA_GOC_DE_THAM_KHAO/`: CSV kết quả **đã có trước**, không phải kết quả mới.
- `KET_QUA_KIEM_THU_FULL/`: CSV, ảnh và metadata từ lần chạy full vừa được kiểm thử trên môi trường hiện tại (đối chiếu trong `DOI_CHIEU_KET_QUA_GOC_MOI.md`).
- `KIEM_TRA_DU_LIEU.py`: kiểm tra đầu vào trước train.
- `requirements.txt` + `.bat`: chạy trên Windows.

## Chạy

1. Giải nén ZIP.
2. Mở Terminal tại thư mục có `.py`.
3. Cài các thư viện: `py -m pip install -r requirements.txt`.
4. Chạy kiểm tra data: `py KIEM_TRA_DU_LIEU.py`.
5. **Chạy thực tế trên 110.704 dòng đủ điều kiện:** `py ADY201m_RQ1_RQ2_RQ3_PIPELINE.py`.
6. Xem `KET_QUA_CHAY_MOI/` để lấy kết quả RQ1–3 từ lần train vừa chạy.

Muốn thử nhanh luồng code mà không phải huấn luyện toàn bộ: `py ADY201m_RQ1_RQ2_RQ3_PIPELINE.py --quick-check`. Kết quả demo nhanh vào `KET_QUA_CHAY_NHANH/` và **không được dùng số liệu này để báo cáo**.

## Phương pháp và tránh rò rỉ

- **RQ1:** Pearson r giữa điểm thành phần pre-final và `TARGET`, tính trong từng nguồn; xếp hạng median |r| theo nhóm thành phần, ưu tiên nhóm có mặt ở ít nhất 5 nguồn. Đây là liên hệ thống kê, không kết luận nhân quả.
- **RQ2:** Chia nhóm (SOURCE_FILE, CODE) thành train/validation/test ~60/20/20 (GroupShuffleSplit, random_state=42); missing values được điền median từ train; 5 model train trên cùng tập train, chọn bằng F2 ở ngưỡng 0,5 trên validation. Đánh giá model được chọn trên test.
- **RQ3:** Hai mô hình baseline và train-only SMOTE được chọn ngưỡng tối đa F2 trên validation, so sánh F2, Recall, Precision, F1 và ma trận nhầm lẫn trên test. CSV `RQ3_four_scenarios.csv` còn phân biệt riêng ảnh hưởng của threshold tuning và SMOTE.
- `FINAL EXAM`, `TOTAL`, `TARGET`, `STATUS` không được sử dụng làm input dự báo. `CODE` và `SOURCE_FILE` chỉ để nhóm split.

## Kết quả và tính toàn vẹn

- Dữ liệu và kết quả có sẵn trong `KET_QUA_GOC_DE_THAM_KHAO/` là **chỉ để đối chiếu**, không được trình bày như đã chạy lại.
- Các giá trị từ `--quick-check` có thể khác bản full vì giảm dòng và n_estimators; chạy full mới dùng để thuyết trình.
- Trong full mode, code giữ `n_estimators=300` (DT ngoại lệ), SMOTE mặc định random_state 42, threshold từ 0,05 đến 0,95 mỗi bước 0,01 đúng logic pipeline GROUPED cũ.
- Tối ưu F2 không đồng nghĩa Recall chắc chắn tăng; phải báo cáo cả F2 và Recall.
- Đây là pipeline RQ/ML trên **dữ liệu đã làm sạch**. Nếu cần chạy lại bước làm sạch từ Excel thô, dùng `clean_data.py` trong bản gốc 104 MB, không thuộc ZIP rút gọn này.
