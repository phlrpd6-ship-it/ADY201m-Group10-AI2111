/* ADY201m - DEMO SQL SERVER (SSMS)
   Yeu cau: da chay 1 LAN file 01_TAO_DATABASE_ADY201m.sql
   (lenh INSERT tao tu 18 CSV, 124791 ban ghi) de tao CSDL, bang, view.
   Trong buoi demo chi mo file 02_DEMO_SQL.sql va chay tung BLOCK.
   SQL chi xu ly truy van/tong hop, KHONG huan luyen mo hinh ML.
*/
USE ADY201m;
GO

/* 1. KIEM TRA DU LIEU GOC - expected 124791 rows / 18 source files */
SELECT COUNT(*) AS total_rows,
       COUNT(DISTINCT source_file) AS source_files,
       COUNT(DISTINCT code) AS distinct_codes
FROM dbo.ews_cleaned;
GO

/* 2. GROUP BY - dem trang thai sau lam sach */
SELECT status_clean, target, COUNT(*) AS n,
       CAST(100.0 * COUNT(*) / (SELECT COUNT(*) FROM dbo.ews_cleaned)
            AS DECIMAL(10,2)) AS percent_of_all
FROM dbo.ews_cleaned
GROUP BY status_clean, target
ORDER BY target, status_clean;
GO

/* 3. RQ1 - diem PRACTICAL EXAM-30 lien he voi ket qua cuoi ky?
   Day la pooled summary trong SQL, khac RQ1 Python la median
   correlation phan tich RIENG theo file. KHONG so sanh truc tiep 2 he so.
*/
SELECT target,
       COUNT(*) AS students_with_score,
       CAST(AVG(CAST(score AS FLOAT)) AS DECIMAL(10,3)) AS avg_practical_exam_30
FROM dbo.v_rq1_prefinal_long
WHERE assessment_component = N'PRACTICAL EXAM-30'
GROUP BY target
ORDER BY target;
GO

/* 4. RQ2 - 110704 records with pre-final scores; show 10 rows with
   PRACTICAL EXAM-30 not null, so the demo table is informative.
   SQL prepares data; 5 models are evaluated in Python, NOT SQL. */
SELECT COUNT(*) AS modeling_ready_rows FROM dbo.v_ml_prefinal_dataset;
SELECT TOP (10)
       source_file, [PRACTICAL EXAM-30], [ASSIGNMENTS-30],
       [PROGRESS TESTS-30], target
FROM dbo.v_ml_prefinal_dataset
WHERE [PRACTICAL EXAM-30] IS NOT NULL
   OR [ASSIGNMENTS-30] IS NOT NULL
   OR [PROGRESS TESTS-30] IS NOT NULL
ORDER BY record_id;
GO

/* 5. RQ3 - kiem tra mat can bang lop (SMOTE lam trong Python) */
SELECT target,
       COUNT(*) AS n,
       CAST(100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS DECIMAL(10,2)) AS pct
FROM dbo.v_ml_prefinal_dataset
GROUP BY target
ORDER BY target;
GO
