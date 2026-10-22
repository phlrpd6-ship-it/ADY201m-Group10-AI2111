# ADY201m EWS – Data Cleaning Pipeline Results (v2.0)

## Pipeline Summary

| Metric | Value |
|---|---|
| Files processed | 18 / 18 |
| Total rows | 124,791 |
| TARGET=0 (Passed) | 92,179 (73.9%) |
| TARGET=1 (Not Passed / Attendance Fail) | 32,612 (26.1%) |
| Distinct CODEs | 15,201 |
| STATUS mapping flags | 0 |
| Exclusion/data-quality flags | 14,087 rows |
| **Modeling-ready dataset rows** | **110,704** |
| Validation checks | **17/17 passed ✓** |

## Validation Results (all passed)

| Check | Status |
|---|---|
| V01: All 18 expected files processed | ✓ |
| V02: Required ID/meta columns exist | ✓ |
| V03: No missing values in ID/STATUS columns | ✓ |
| V04: Assessment scores within 0-10 range | ✓ |
| V05: TARGET contains only 0/1 | ✓ |
| V06: STATUS_CLEAN → TARGET mapping consistent | ✓ |
| V07: FINAL/TOTAL present for reference (not predictors) | ✓ |
| V08: No accidental duplicate rows | ✓ |
| V09: Source-file traceability preserved | ✓ |
| V10: Row count reconciliation (124,791 = 124,791) | ✓ |
| V11: Diagnostic flag columns present | ✓ |
| V12: 120102BAA rows flagged (11,476 rows) | ✓ |
| V13: Attendance Fail preserved (10,085 rows) | ✓ |
| V14: Q6 verification: 799 contradictions, 89.9% explained by zero-component rule | ✓ |
| V15: Modeling safety: 21 features, no leakage | ✓ |
| V16: Attendance Fail correctly assigned TARGET=1 (10,085 rows) | ✓ |
| V17: QUIZ (2) and PROGRESS TEST (3) remain separate | ✓ |

## Audit Findings Verified Against Data

Key findings from the audit report were **independently verified** against the Excel files before implementing the pipeline:

| Audit Claim | Verified | Result |
|---|---|---|
| 120102BAA has 0 score columns | ✓ | Confirmed: columns are only ID/META/TOTAL |
| 120102BAA TOTAL: 388 non-null, all 0.0 | ✓ | Confirmed |
| STATUS_CLEAN ↔ TARGET mapping is consistent | ✓ | 0 inconsistencies in 124,791 rows |
| 8 truncated STATUS spellings exist | ✓ | Confirmed: Passe, Pass, Not Passe, Attendance Fai, Attendance |
| All scores in 0-10 range | ✓ | Confirmed |
| 15,201 distinct CODEs | ✓ | Confirmed |
| S3: 3,103 rows with TOTAL>0 but all scores=0 | ✓ | Exact match |
| S4: 65 rows with TOTAL=0 but component>0 | ✓ | Exact match |

## Output Files

### Cleaned Data ([data/cleaned/](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/data/cleaned))
- 18 per-file CSVs + 1 combined CSV (19.6 MB)
- [pipeline_metadata.json](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/data/cleaned/pipeline_metadata.json)

### Modeling Dataset ([data/modeling_eligible/](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/data/modeling_eligible))
- [ADY201m_modeling_ready.csv](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/data/modeling_eligible/ADY201m_modeling_ready.csv) — 110,704 rows
- [feature_info.json](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/data/modeling_eligible/feature_info.json)

### Reports
- [Missingness report](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/reports/missingness/missingness_report.csv) — 59 entries
- [Schema report](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/reports/schema/schema_report.csv) — 18 entries
- [Exclusion flags](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/reports/exclusions/exclusion_flags.csv) — 14,087 flagged rows
- [CODE repetition](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/reports/quality/code_repetition_report.csv) — per-file + cross-file + within-semester
- [TOTAL anomalies](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/reports/quality/total_anomaly_diagnostic.csv) — S3/S4 diagnostics
- [Q6 verification](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/reports/quality/q6_verification.csv) — Zero-component rule verification
- [Validation results](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/reports/validation/validation_summary.md) — JSON + CSV + markdown
- [Methodological Decisions](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/reports/methodological_decisions.md) — Log of Q1-Q8 resolutions

### Source Code
- [config.py](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/src/config.py) — Central configuration
- [clean_data.py](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/src/clean_data.py) — Main cleaning pipeline (v2.0)
- [validate_data.py](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/src/validate_data.py) — Post-cleaning validation (v2.0)
- [README.md](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/README.md) — Full documentation

## Flags Added to Cleaned Data

| Flag Column | True Count | Purpose |
|---|---|---|
| `HAS_PREFINAL` | 110,704 | Has at least one non-FINAL assessment |
| `HAS_PREFINAL_EXCL_PE` | 110,704 | Excludes PRACTICAL EXAM / PROJECT (retained for diagnostics) |
| `NO_ASSESSMENT_DATA` | 14,087 | All score columns NaN |
| `FLAG_120102BAA` | 11,476 | From file with no assessment predictors |
| `FLAG_SUSPICIOUS_TOTAL` | 388 | TOTAL is a known placeholder |
| `FLAG_NO_SCORE_ERA` | 14,073 | From semester with no score data |

*(Note: `FLAG_TIMING_UNCERTAIN` was removed in v2.0 after resolving Q1)*

## Methodological Decisions (Resolved)

All previously unresolved questions have been formally resolved in v2.0. See [`reports/methodological_decisions.md`](file:///c:/PHAMHOANGLONG/HOC/ADY/ADY201m/reports/methodological_decisions.md) for full rationale:

1. **Q1**: PRACTICAL EXAM and PROJECT are confirmed pre-final and included as predictors.
2. **Q2**: Column name weights are unverified; scores are kept as raw values.
3. **Q3**: CODE is used for within-file traceability only, not ML features.
4. **Q4**: QUIZ and PROGRESS TEST are kept as separate features.
5. **Q5**: Attendance Fail is kept as TARGET=1 (unsuccessful outcome).
6. **Q6**: The 799 anomalous rows are largely explained by the zero-component failure rule.
7. **Q7**: Pooled modeling will be the primary approach.
8. **Q8**: The 593 upstream exclusions remain unauditable (documented limitation).

## How to Reproduce

```bash
cd c:\PHAMHOANGLONG\HOC\ADY
py ADY201m\src\setup_project.py    # Set up directory structure
py ADY201m\src\clean_data.py       # Run cleaning pipeline
py ADY201m\src\validate_data.py    # Validate outputs
```
