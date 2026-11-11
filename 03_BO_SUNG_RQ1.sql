/* ======================================================================
 ADY201m - SQL Server / SSMS - RQ1 supplementary queries (add-on to 02)
 Run in the SAME query window right after 02, because Q-E reads the temp
 table #rq1_per_component created by 02. Read-only: no tables are changed.
 NOT yet executed in SSMS by the author of this draft - run and check.
 ====================================================================== */
USE ADY201m;
GO

/* Q-A [Data Understanding + RQ1 denominator]
   Rows, modeling-ready rows and unsuccessful rate per source file. */
SELECT source_file,
       COUNT_BIG(*) AS total_rows,
       SUM(CASE WHEN has_prefinal = 1 THEN 1 ELSE 0 END) AS modeling_ready_rows,
       CAST(100.0*SUM(CASE WHEN has_prefinal = 1 THEN 1 ELSE 0 END)/COUNT(*) AS DECIMAL(5,2)) AS pct_modeling_ready,
       CAST(100.0*SUM(CASE WHEN has_prefinal = 1 AND target = 1 THEN 1 ELSE 0 END)
            / NULLIF(SUM(CASE WHEN has_prefinal = 1 THEN 1 ELSE 0 END),0) AS DECIMAL(5,2)) AS pct_unsuccessful
FROM dbo.ews_cleaned
GROUP BY source_file
ORDER BY source_file;
GO

/* Q-B [Coverage of each component inside its file]
   Shows why some families rest on few files / few rows (no imputation). */
;WITH f AS (
 SELECT source_file, COUNT_BIG(*) AS file_rows
 FROM dbo.ews_cleaned WHERE has_prefinal = 1
 GROUP BY source_file
)
SELECT l.source_file, l.assessment_component,
       COUNT_BIG(*) AS n_scored, f.file_rows,
       CAST(100.0*COUNT_BIG(*)/f.file_rows AS DECIMAL(5,2)) AS pct_scored
FROM dbo.v_rq1_prefinal_long AS l
JOIN f ON f.source_file = l.source_file
GROUP BY l.source_file, l.assessment_component, f.file_rows
ORDER BY l.source_file, pct_scored DESC;
GO

/* Q-C [Evidence for the repeated-record issue]
   Key = (source_file, code, semester). Report as-is; no dedup (run A). */
;WITH d AS (
 SELECT source_file, code, semester, COUNT_BIG(*) AS c
 FROM dbo.ews_cleaned
 GROUP BY source_file, code, semester
 HAVING COUNT_BIG(*) > 1
)
SELECT COUNT_BIG(*) AS repeated_keys,
       SUM(c)       AS rows_in_repeated_keys,
       SUM(c - 1)   AS extra_rows
FROM d;
GO

/* Q-D0 [Sanity check BEFORE Q-D1/Q-D2]
   Is score on the 0..weight scale? If max_score is far from weight
   (e.g. everything <= 10 for a -30 component), rescale in Q-D2. */
SELECT assessment_component,
       CAST(RIGHT(assessment_component, CHARINDEX('-', REVERSE(assessment_component)) - 1) AS INT) AS weight,
       MIN(score) AS min_score, MAX(score) AS max_score, COUNT_BIG(*) AS n
FROM dbo.v_rq1_prefinal_long
GROUP BY assessment_component
ORDER BY assessment_component;
GO

/* Q-D1 [Direction + effect size for the answer family]
   Mean score of successful (0) vs unsuccessful (1) per file x component.
   Change @family to inspect another family (e.g. N'PROJECT'). */
DECLARE @family NVARCHAR(100) = N'PRACTICAL EXAM';

SELECT source_file, assessment_component,
       COUNT_BIG(*) AS n,
       CAST(AVG(CASE WHEN target = 0 THEN CAST(score AS FLOAT) END) AS DECIMAL(10,3)) AS mean_score_successful,
       CAST(AVG(CASE WHEN target = 1 THEN CAST(score AS FLOAT) END) AS DECIMAL(10,3)) AS mean_score_unsuccessful,
       CAST(AVG(CASE WHEN target = 0 THEN CAST(score AS FLOAT) END)
          - AVG(CASE WHEN target = 1 THEN CAST(score AS FLOAT) END) AS DECIMAL(10,3)) AS mean_gap
FROM dbo.v_rq1_prefinal_long
WHERE assessment_component LIKE @family + N'-%'
GROUP BY source_file, assessment_component
HAVING COUNT_BIG(*) >= 30
ORDER BY mean_gap DESC;
GO


/* Q-D2 [Unsuccessful rate by score band, pooled over the family]
   Scores are recorded on a 0-10 scale.
   Band = percentage of maximum attainable score. */

DECLARE @family NVARCHAR(100) = N'PRACTICAL EXAM';

;WITH b AS (
    SELECT
        l.target,
        100.0 * l.score / 10.0 AS pct_of_max
    FROM dbo.v_rq1_prefinal_long AS l
    WHERE l.assessment_component LIKE @family + N'-%'
      AND l.score IS NOT NULL
),
bb AS (
    SELECT
        target,
        CASE
            WHEN pct_of_max < 25 THEN N'1) 0-25%'
            WHEN pct_of_max < 50 THEN N'2) 25-50%'
            WHEN pct_of_max < 75 THEN N'3) 50-75%'
            ELSE N'4) 75-100%'
        END AS score_band
    FROM b
    WHERE pct_of_max BETWEEN 0 AND 100
)
SELECT
    score_band,
    COUNT_BIG(*) AS n,
    SUM(CAST(target AS INT)) AS n_unsuccessful,
    CAST(
        100.0 * SUM(CAST(target AS INT)) / COUNT_BIG(*)
        AS DECIMAL(5,2)
    ) AS pct_unsuccessful
FROM bb
GROUP BY score_band
ORDER BY score_band;
GO


/* Rebuild the temporary table required by Q-E */

DROP TABLE IF EXISTS #rq1_per_component;

;WITH S AS (
    SELECT
        source_file,
        assessment_component,
        COUNT_BIG(*) AS n,
        SUM(CAST(score AS FLOAT)) AS sx,
        SUM(CAST(target AS FLOAT)) AS sy,
        SUM(CAST(score AS FLOAT) * CAST(target AS FLOAT)) AS sxy,
        SUM(CAST(score AS FLOAT) * CAST(score AS FLOAT)) AS sx2,
        SUM(CAST(target AS FLOAT) * CAST(target AS FLOAT)) AS sy2
    FROM dbo.v_rq1_prefinal_long
    GROUP BY source_file, assessment_component
    HAVING COUNT_BIG(*) >= 30
),
R AS (
    SELECT
        source_file,
        assessment_component,
        n,
        (CAST(n AS FLOAT) * sxy - sx * sy) /
        NULLIF(
            SQRT(
                (CAST(n AS FLOAT) * sx2 - sx * sx) *
                (CAST(n AS FLOAT) * sy2 - sy * sy)
            ), 0
        ) AS pearson_r
    FROM S
    WHERE
        (CAST(n AS FLOAT) * sx2 - sx * sx) > 0
        AND
        (CAST(n AS FLOAT) * sy2 - sy * sy) > 0
)
SELECT
    source_file,
    assessment_component,
    n,
    pearson_r,
    LEFT(
        assessment_component,
        LEN(assessment_component)
        - CHARINDEX('-', REVERSE(assessment_component))
    ) AS assessment_family
INTO #rq1_per_component
FROM R
WHERE pearson_r IS NOT NULL;
GO

/* Q-E [Accepted sensitivity analysis]
   median within (file, family) first, then median across files.
   Needs #rq1_per_component from 02 (same session). Compare ranking
   with RQ1-B: if the top families are unchanged, RQ1 is robust. */
;WITH ff AS (
 SELECT DISTINCT source_file, assessment_family,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(pearson_r))
          OVER (PARTITION BY source_file, assessment_family) AS file_med_abs_r
 FROM #rq1_per_component
), fam AS (
 SELECT assessment_family, source_file, file_med_abs_r,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY file_med_abs_r)
          OVER (PARTITION BY assessment_family) AS med_of_file_med
 FROM ff
)
SELECT ROW_NUMBER() OVER (ORDER BY MAX(med_of_file_med) DESC) AS rank_sensitivity,
       assessment_family,
       COUNT(*) AS distinct_files,
       CAST(MAX(med_of_file_med) AS DECIMAL(12,6)) AS median_of_file_medians
FROM fam
GROUP BY assessment_family
ORDER BY rank_sensitivity;
GO

SELECT
    COUNT_BIG(*) AS total_rows,
    COUNT_BIG(l.score) AS nonnull_scores,
    COUNT_BIG(l.target) AS nonnull_targets,
    COUNT_BIG(
        CASE
            WHEN l.score IS NOT NULL
             AND l.target IS NOT NULL
            THEN 1
        END
    ) AS valid_score_target_pairs,
    COUNT_BIG(
        CASE
            WHEN l.score IS NOT NULL
             AND l.target IS NULL
            THEN 1
        END
    ) AS missing_targets
FROM dbo.v_rq1_prefinal_long AS l
WHERE l.assessment_component LIKE N'PRACTICAL EXAM-%';

