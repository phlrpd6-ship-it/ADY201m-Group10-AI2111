/* ======================================================================
 ADY201m - SQL Server / SSMS - RQ1 / RQ2 / RQ3 (dong bo FULL RUN)
 INPUT : 18 cleaned CSV tuong ung 01_TAO_DATABASE_ADY201m.sql
 METRICS: KET_QUA_CHAY_V1_1 / pipeline v1.1 FULL run 2026-11-04 (khong imputation, tru nhanh SMOTE)
 RQ1: SQL calculates Pearson within each SOURCE_FILE and component.
 RQ2/RQ3: SQL queries saved prediction metrics from Python pipeline FULL;
           DOES NOT TRAIN models or APPLY SMOTE.
 Target: 1=unsuccessful, 0=successful.
 WARNING: run 01 only on your practice database: it recreates its own tables.
 ====================================================================== */
USE ADY201m;
GO

/* QUERY 0: Input data verification */
SELECT COUNT_BIG(*) AS total_rows, COUNT(DISTINCT source_file) AS files,
       COUNT(DISTINCT code) AS distinct_code_values,
       SUM(CAST(has_prefinal AS BIGINT)) AS eligible_rows
FROM dbo.ews_cleaned;
-- Expected 124791, 18, 15201, 110704.
GO

/* ======================================================================
 RQ1: Which pre-final assessment components are most strongly associated
      with an unsuccessful outcome?
 Comparability note: 18 source files have different assessment structures.
 For each (file, component) with >=30 paired rows, calculate point-biserial
 Pearson correlation with binary target. Then aggregate to family-level
 MEDIAN(abs(r)). Present coverage to avoid overclaiming sparse families.
 ====================================================================== */
DROP TABLE IF EXISTS #rq1_per_component;
;WITH S AS (
 SELECT source_file, assessment_component, COUNT_BIG(*) AS n,
        SUM(CAST(score AS FLOAT)) AS sx,
        SUM(CAST(target AS FLOAT)) AS sy,
        SUM(CAST(score AS FLOAT)*CAST(target AS FLOAT)) AS sxy,
        SUM(CAST(score AS FLOAT)*CAST(score AS FLOAT)) AS sx2,
        SUM(CAST(target AS FLOAT)*CAST(target AS FLOAT)) AS sy2
 FROM dbo.v_rq1_prefinal_long
 GROUP BY source_file, assessment_component
 HAVING COUNT_BIG(*) >= 30
), R AS (
 SELECT source_file,assessment_component,n,
        (CAST(n AS FLOAT)*sxy-sx*sy)/
        NULLIF(SQRT((CAST(n AS FLOAT)*sx2-sx*sx)*
                    (CAST(n AS FLOAT)*sy2-sy*sy)),0) AS pearson_r
 FROM S
 WHERE (CAST(n AS FLOAT)*sx2-sx*sx)>0 AND (CAST(n AS FLOAT)*sy2-sy*sy)>0
)
SELECT source_file, assessment_component, n, pearson_r,
       LEFT(assessment_component,LEN(assessment_component)-CHARINDEX('-',REVERSE(assessment_component))) AS assessment_family
INTO #rq1_per_component
FROM R WHERE pearson_r IS NOT NULL;

/* QUERY RQ1-A: component-level measured evidence */
SELECT source_file,assessment_component,n,
       CAST(pearson_r AS DECIMAL(12,6)) AS pearson_r,
       CAST(ABS(pearson_r) AS DECIMAL(12,6)) AS abs_r
FROM #rq1_per_component ORDER BY ABS(pearson_r) DESC;

DROP TABLE IF EXISTS #rq1_families;
;WITH M AS (
 SELECT *,PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(pearson_r))
        OVER (PARTITION BY assessment_family) AS med_abs_r
 FROM #rq1_per_component
)
SELECT assessment_family, COUNT(DISTINCT source_file) AS distinct_files,
       COUNT(*) AS component_file_pairs, SUM(n) AS valid_pairs,
       CAST(MAX(med_abs_r) AS DECIMAL(12,6)) AS median_abs_r
INTO #rq1_families FROM M GROUP BY assessment_family;

/* QUERY RQ1-B: rank all families; SHOW coverage */
SELECT ROW_NUMBER() OVER (ORDER BY median_abs_r DESC) AS rank_overall,
       assessment_family, distinct_files, component_file_pairs, valid_pairs,
       median_abs_r
FROM #rq1_families ORDER BY median_abs_r DESC;

/* QUERY RQ1-C: best broadly represented family (>=5 source files) */
SELECT TOP(1) assessment_family AS rq1_answer,
       distinct_files,median_abs_r
FROM #rq1_families WHERE distinct_files>=5
ORDER BY median_abs_r DESC;
-- Expected PRACTICAL EXAM, 10 files, median |r|~0.719285.
-- PROJECT ~0.766637 but only 2 source files; report this limitation.
GO

/* ======================================================================
 RQ2: Which of 5 algorithms performs best for the early-warning task?
 Metrics copied from FULL pipeline output RQ2_model_comparison.csv, NOT trained
 by SQL. Chosen by validation F2 at 0.50; test result only for chosen model.
 ====================================================================== */
DROP TABLE IF EXISTS #rq2_models;
CREATE TABLE #rq2_models (
 model NVARCHAR(70), val_tp INT,val_fp INT,val_fn INT,
 val_f2_saved FLOAT, val_pr_auc FLOAT,test_f2_saved FLOAT NULL
);
INSERT INTO #rq2_models VALUES
    (N'Random Forest', 5040, 897, 830, 0.8566475167420199, 0.930896859932578, 0.8408683108749956),
    (N'LightGBM', 4850, 283, 1020, 0.8475168629643868, 0.9463992272532308, NULL),
    (N'XGBoost', 4833, 253, 1037, 0.8459357277882797, 0.9464196447908712, NULL),
    (N'CatBoost', 4780, 239, 1090, 0.8386259166988316, 0.9444055807433444, NULL),
    (N'Decision Tree', 4953, 1140, 917, 0.8374192675751531, 0.783516494696516, NULL);

/* QUERY RQ2-A: recalculate F2 from confusion matrix for validation */
SELECT ROW_NUMBER() OVER (ORDER BY (5.0*val_tp)/(5.0*val_tp+4.0*val_fn+val_fp) DESC) AS rank_by_f2,
       model,
       CAST(1.0*val_tp/NULLIF(val_tp+val_fp,0) AS DECIMAL(10,6)) AS val_precision,
       CAST(1.0*val_tp/NULLIF(val_tp+val_fn,0) AS DECIMAL(10,6)) AS val_recall,
       CAST((5.0*val_tp)/(5.0*val_tp+4.0*val_fn+val_fp) AS DECIMAL(10,6)) AS val_f2_recalculated,
       CAST(ABS((5.0*val_tp)/(5.0*val_tp+4.0*val_fn+val_fp)-val_f2_saved) AS DECIMAL(12,9)) AS f2_check_error,
       CAST(val_pr_auc AS DECIMAL(10,6)) AS validation_pr_auc,
       CAST(test_f2_saved AS DECIMAL(10,6)) AS test_f2_when_available
FROM #rq2_models ORDER BY val_f2_recalculated DESC;

/* QUERY RQ2-B: direct answer (do not select by test metrics!) */
SELECT TOP(1) model AS rq2_best_model,
       CAST((5.0*val_tp)/(5.0*val_tp+4.0*val_fn+val_fp) AS DECIMAL(10,6)) AS validation_f2
FROM #rq2_models
ORDER BY (5.0*val_tp)/(5.0*val_tp+4.0*val_fn+val_fp) DESC;
-- Expected: Random Forest / ~0.856648 (chenh hang 2 chi ~0.0091; 1 lan chia, 1 random_state).
GO

/* ======================================================================
 RQ3: Does SMOTE + F2-threshold tuning improve recall-oriented performance?
 Four test scenarios permit a controlled, fair comparison.
 Tuning happens ONLY on validation; SMOTE ONLY on training fold in Python. v1.1: no imputation except the SMOTE branch.
 ====================================================================== */
/* QUERY RQ3-A: actual class distribution BEFORE SMOTE */
SELECT target,COUNT_BIG(*) AS n,
       CAST(100.0*COUNT_BIG(*)/SUM(COUNT_BIG(*)) OVER() AS DECIMAL(10,2)) AS pct
FROM dbo.v_ml_prefinal_dataset GROUP BY target ORDER BY target;

DROP TABLE IF EXISTS #rq3_four;
CREATE TABLE #rq3_four (
 scenario NVARCHAR(80), threshold FLOAT,
 tn INT,fp INT,fn INT,tp INT,
 f2_saved FLOAT, recall_saved FLOAT, precision_saved FLOAT
);
INSERT INTO #rq3_four VALUES
    (N'Baseline @ 0.50', 0.5, 15407, 879, 911, 4780, 0.8408683108749956, 0.8399226849411351, 0.8446722035695352),
    (N'Baseline + F2 tuning', 0.39, 14843, 1443, 801, 4890, 0.8402928136921332, 0.8592514496573537, 0.7721459024159166),
    (N'SMOTE @ 0.50', 0.5, 15607, 679, 987, 4704, 0.835613031584183, 0.8265682656826568, 0.8738621586475943),
    (N'SMOTE + F2 tuning', 0.34, 15029, 1257, 818, 4873, 0.8432546549456634, 0.8562642769284836, 0.7949429037520391);

/* QUERY RQ3-B: recalculate TEST metrics, not copied summary only */
SELECT scenario,CAST(threshold AS DECIMAL(5,2)) AS threshold,
       tn,fp,fn,tp,
       CAST(1.0*tp/NULLIF(tp+fp,0) AS DECIMAL(10,6)) AS test_precision,
       CAST(1.0*tp/NULLIF(tp+fn,0) AS DECIMAL(10,6)) AS test_recall,
       CAST((5.0*tp)/(5.0*tp+4.0*fn+fp) AS DECIMAL(10,6)) AS test_f2,
       CAST(ABS((5.0*tp)/(5.0*tp+4.0*fn+fp)-f2_saved) AS DECIMAL(12,9)) AS check_error
FROM #rq3_four ORDER BY
 CASE scenario WHEN N'Baseline @ 0.50' THEN 1
               WHEN N'Baseline + F2 tuning' THEN 2
               WHEN N'SMOTE @ 0.50' THEN 3
               WHEN N'SMOTE + F2 tuning' THEN 4 END;

/* QUERY RQ3-C: FAIR question: tuned SMOTE versus tuned baseline */
;WITH m AS (
 SELECT scenario,threshold,
        5.0*tp/(5.0*tp+4.0*fn+fp) AS f2,
        1.0*tp/(tp+fn) AS recall,
        1.0*tp/(tp+fp) AS precision
 FROM #rq3_four
)
SELECT CAST(b.f2 AS DECIMAL(10,6)) AS baseline_tuned_f2,
       CAST(s.f2 AS DECIMAL(10,6)) AS smote_tuned_f2,
       CAST(s.f2-b.f2 AS DECIMAL(12,6)) AS delta_test_f2,
       CAST(s.recall-b.recall AS DECIMAL(12,6)) AS delta_test_recall,
       CAST(s.precision-b.precision AS DECIMAL(12,6)) AS delta_test_precision,
       CASE WHEN ABS(s.f2-b.f2)<0.005 THEN N'Negligible F2 difference (descriptive, single split): no evidence SMOTE improves F2'
            WHEN s.f2>b.f2 THEN N'SMOTE+tuning higher F2 (descriptive, single split)'
            ELSE N'SMOTE+tuning lower F2 (descriptive, single split)' END AS rq3_answer
FROM m b CROSS JOIN m s
WHERE b.scenario=N'Baseline + F2 tuning'
  AND s.scenario=N'SMOTE + F2 tuning';
-- Expected ~0.840293 -> ~0.843255, delta +0.002962; recall delta -0.002987.

/* QUERY RQ3-D: effect of threshold tuning WITHOUT SMOTE */
;WITH M AS (
 SELECT scenario,5.0*tp/(5.0*tp+4.0*fn+fp) AS f2
 FROM #rq3_four
)
SELECT CAST(t.f2-u.f2 AS DECIMAL(12,6)) AS delta_f2_tuning_only
FROM M u CROSS JOIN M t
WHERE u.scenario=N'Baseline @ 0.50'
  AND t.scenario=N'Baseline + F2 tuning';
-- Separates threshold gain from SMOTE effect.
GO
