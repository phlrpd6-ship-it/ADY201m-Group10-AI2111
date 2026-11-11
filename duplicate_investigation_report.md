# Investigation Report: Repeated `(SOURCE_FILE, CODE, Mã học kỳ)` Records in the ADY201m EWS Dataset

**Version 2: full-data verification.** Supersedes my earlier partial version. This run used the 18 raw workbooks, the 18 course CSVs, `ADY201m_combined_cleaned.csv`, and `pipeline_metadata.json`. All inputs were read-only; nothing was modified. Evidence labels: **[Demonstrated]**, **[Plausible]**, **[Not established]**.

## 1. Executive summary

1. **The count reproduces from the combined CSV** [Demonstrated]: 4,812 repeated combinations (4,811 pairs, 1 triple), 9,625 rows, 4,813 excess rows. No null keys.
2. **The repeats are in the original Excel files; cleaning did not create them** [Demonstrated]. Every one of the 124,791 cleaned rows pairs exactly with a raw row on all raw columns (0 changed values). All 9,625 repeated records trace to distinct raw rows, in all 12 affected files including `161810CJ24`. The combined CSV equals the concatenation of the 18 course CSVs.
3. **Cleaning removed repeats, not added them** [Demonstrated]. Raw has 4,927 repeated combinations among rows with a CODE. The pipeline removed 593 raw rows (503 "Is Susspended", 55 "Is Exempt", 35 blank rows). That collapsed 115 combinations to a single row, leaving exactly 4,812.
4. **No repeat is an exact duplicate.** All 4,812 combinations contain different `Mã lớp` values; in 4,449 of 4,811 pairs at least one score column present in both records differs.
5. **Labels conflict widely:** 3,317 combinations (68.9%) contain both `TARGET=0` and `1`. The `STATUS`→`STATUS_CLEAN`→`TARGET` mapping itself is exact (no exceptions), so this conflict comes from the source data, not mis-mapping.
6. **Why the source contains these repeats is not established.** Concurrent enrolments, transfers, retakes and system artefacts all remain possible; nothing in the data separates them.
7. **For modelling:** 9,033 of the 110,704 modelling rows (8.2%) belong to repeated combinations; 3,123 combinations with ≥2 modelling rows have conflicting labels.

## 2. Dataset and file provenance

| Item | Result |
|---|---|
| Files | 18 raw `.xlsx` (sheet `Data`), 18 course CSVs, 1 combined CSV, `pipeline_metadata.json`, modelling pipeline script |
| Row counts | Raw 125,384 → cleaned 124,791 (= combined = sum of course CSVs = `total_rows` in metadata) |
| Removed rows | 593 across 15 files (none in `091520AJA`, `160316BIA`, `161806AIC`); full list with raw Excel row numbers in `full_outputs/raw_rows_removed_by_pipeline.csv` |
| Unchanged values | All cleaned rows equal their raw rows on every raw column. Cleaning added derived columns (`STATUS_CLEAN`, `TARGET`, flags, `ROW_IDX_ORIG`) |
| `ROW_IDX_ORIG` | Contiguous 0..n-1 per cleaned file, assigned *after* removals; **not** the raw row position. Use `cleaned_to_raw_row_map.csv` (`RAW_EXCEL_ROW` = pandas index + 2, header on row 1) |
| Gap in provenance | The cleaning metadata shows its input was a `partially_cleaned` folder, not the raw folder. That earlier step is not documented in what I received; I can only say that the raw workbooks supplied are consistent with the final data |

## 3. Reproduction of the repeated-record count

| Metric | Value |
|---|---|
| Distinct repeated combinations | 4,812 |
| Rows involved | 9,625 (7.7% of 124,791) |
| Excess rows | 4,813 |
| Frequency | 2 occurrences: 4,811; 3 occurrences: 1 |

## 4. Statistical breakdown

| File | Repeated combos | Rows in repeats | % of file's rows |
|---|---:|---:|---:|
| 030501BJA | 100 | 200 | 1.3% |
| 031904BJA | 402 | 805 | 6.8% |
| 031904BJC | 52 | 104 | 7.9% |
| 040209BJB | 374 | 748 | 5.3% |
| 091520AJB | 13 | 26 | 0.3% |
| 120102BAA | 203 | 406 | 3.5% |
| 151907BJB | 69 | 138 | 1.0% |
| 161806AIB | 987 | 1,974 | 11.9% |
| 161810CBA | 278 | 556 | 22.9% |
| 161810CJ24 | 497 | 994 | 20.0% |
| 161810CJA | 787 | 1,574 | 25.8% |
| 161815AIB | 1,050 | 2,100 | 14.0% |

No repeats in `010425BJA13`, `040116CIA13`, `091520AJA`, `160316BIA`, `160616AIA`, `161806AIC`.

| Content pattern | Combinations |
|---|---:|
| Scores differ **and** outcome labels differ | 3,513 |
| Scores differ, labels agree | 951 |
| Outcome labels differ, no score difference | 162 |
| Only `Mã lớp` (and sometimes `NO.`) differ | 186 |
| Exact duplicates | **0** |

The pattern table was derived from OpenWork's detail table, which I confirmed holds exactly the 9,625 repeat rows of the real combined CSV with identical values (all 33 numeric columns equal). A NaN-aware recheck changes no category.

Other facts [Demonstrated]:
- `Mã lớp` differs in 100% of combinations; `NO.` differs in 4,187 of 4,811 pairs.
- Label pairs: Not Passed ∥ Passed 2,820; Attendance Fail ∥ Passed 497; Attendance Fail ∥ Not Passed 358; same label in both 1,137.
- Pair members are never adjacent (minimum gap 12 rows; median 260).
- Only 14 pairs show one-sided missing scores; the repeats are not one enrolment split across partial rows.
- Class numbering: in Not Passed/Passed pairs the Passed record has the lower class number 2,676 times and the higher 640 times (same split as file order, so file order follows class order). A retake-after-failure story would predict the opposite; this argues against it but does not rule out other explanations.

## 5. Representative record comparisons

| Pattern | File / CODE / semester | Record A | Record B |
|---|---|---|---|
| Conflicting assessment | 030501BJA, 1905ACJCJJ, FA2020 | class 2, NO. 5, FINAL 0.0, TOTAL 3.6, Not Passed (cleaned row 853) | class 3, NO. 3, FINAL 0.0, TOTAL 0.3, Not Passed (row 867) |
| Assessment + outcome | 030501BJA, 1905ADJGBC, FA2020 | class 2, NO. 8, FINAL 4.0, TOTAL 6.7, Passed, TARGET 0 (row 856) | class 4, NO. 15, FINAL 0.0, TOTAL 1.7, Attendance Fail, TARGET 1 (row 907) |
| Outcome only | 120102BAA, 1905ACJBFF, FA2018 | class 3, NO. 1, no scores, Not Passed (row 190) | class 7, NO. 1, no scores, Passed (row 300) |
| Class label only | 030501BJA, 1905ACJEBE, SP2020 | class 4, NO. 11, TOTAL 0.0, Attendance Fail (row 2892) | class 5, NO. 16, TOTAL 0.0, Attendance Fail (row 2915) |

Cleaned-row numbers are CSV data rows; raw Excel rows for every record are in `duplicate_record_details_with_raw_rows.csv`.

## 6. Original-versus-cleaned source verification

**Method.** Each raw row was reduced to the tuple of its normalised raw-column values; each cleaned row was reduced to the same tuple and paired to a raw row (identical tuples paired in file order). This replaces OpenWork's position-based comparison, which was invalid: cleaned indices restart at 0 after removals, so cleaned row *i* is not raw row *i* (its "8,870 values differ" was a misalignment artefact).

**Results** [Demonstrated]:
- 124,791 of 124,791 cleaned rows pair to a raw row; 0 cleaned rows lack an exact raw match; 593 raw rows have no cleaned partner.
- 9,625 of 9,625 repeated records pair to a raw row; the members of each of the 4,812 combinations map to distinct raw rows. This includes the 497 `161810CJ24` combinations that OpenWork could not match (the earlier "ambiguous" label was really "no match" under its strict key; I did not diagnose why, and value-based pairing makes it moot).
- The 593 removed rows: 502 `Is Susspended`, 1 `Is Susspende` (typo), 55 `Is Exempt`, 35 with blank CODE and STATUS. None share (CODE, semester, class) with a kept row; 115 share (CODE, semester) with a kept row, and these are the 115 combinations that became singletons (114 suspended, 1 exempt).
- Status typos exist in the raw data (`Passe`, `Not Passe`, `Attendance Fai`, `Pass`, `Attendance`: 8 rows) and are corrected in `STATUS_CLEAN`; `TARGET` is 1 for Not Passed and Attendance Fail, 0 for Passed, with no exceptions.

**Answer to the task's question:** the repeated records and their conflicting values already existed in the original workbooks; the cleaning process did not duplicate, transform or mis-map them.

## 7. Evidence-based classification

| Category | Combinations | Evidence |
|---|---:|---|
| Repeat present in source, values unchanged by cleaning | 4,812 (all) | Demonstrated |
| Exact duplicate records | 0 | Demonstrated |
| Conflicting assessment records | 4,464 (951 + 3,513) | Demonstrated |
| Conflicting outcome labels | 3,675 (162 + 3,513) differ in status; 3,317 span both TARGET values | Demonstrated |
| Class label differs, scores/outcome otherwise identical | 186, but 133 have TOTAL 0/blank so "identical" is uninformative; only **53** carry identical non-zero totals (38 Passed, 11 Not Passed, 4 Attendance Fail) | Demonstrated; those 53 are the best candidates for re-labelling or duplication, which remains Plausible only |
| Outcome-only conflicts | 162, of which 149 have no assessment data (mainly `120102BAA`) | Label differences can't be checked against any scores |
| Legitimate separate enrolments | Not established for any | Not established |
| Class transfers / administrative changes / retakes | Not established | Plausible only; different `Mã lớp` fits it but is not proof, and class order argues against a simple retake pattern |

The honest label for most repeats (different class, different scores, different outcome) is **real source rows that this dataset cannot disambiguate**.

## 8. Implications for the EWS methodology

- **Target reliability:** 68.9% of repeated combinations hold both labels. Keeping one row (first, last, best) flips labels arbitrarily; keeping both counts one student-course-semester twice with opposite labels.
- **Assessment analysis:** per-student aggregates or joins on `(CODE, semester)` mix or fan out scores unless `Mã lớp` is part of the key.
- **Modelling set:** 9,033 of 110,704 rows (8.2%) are in repeated combinations; 3,123 of the 4,516 combinations with ≥2 modelling rows are label-conflicting. Those rows are not independent observations and will cap achievable accuracy.
- **Grouped split by `(SOURCE_FILE, CODE)`** (as in your pipeline): every repeated combination shares its `(SOURCE_FILE, CODE)`, so the group split keeps all repeated rows of a code within one course in the same fold, including across semesters. In the modelling data, 4,320 such groups contain repeated rows. **It does not resolve conflicts** (rows stay together but still disagree) **and does not establish global student identity**: 4,788 of the 4,812 combinations involve a CODE that also appears in 2+ course files; whether `CODE` is the same person across courses is untested, and if it is, cross-course leakage across folds remains.

## 9. Limitations and unresolved questions

1. Why the source files contain these repeats (concurrent classes, transfers, retakes, system artefacts) is undetermined; it needs the registrar or whoever produced the workbooks.
2. The `raw → partially_cleaned` step is undocumented; I verified consistency with the raw workbooks supplied, not that these are the true originals.
3. `CODE` identity across courses is untested.
4. The removal rule for `Is Susspended` / `Is Exempt` is visible in the data but its rationale lives in the pipeline, which I did not review.
5. Not examined: the 799 status/score contradictions listed in the metadata (`q6_verification`).

## 10. Recommendations for the research report

1. State that all 4,812 repeated combinations exist in the source workbooks, unchanged by cleaning, and that their meaning is undetermined.
2. Report conflict rates (68.9% label conflict; ~93% score difference) and the 8.2% share of modelling rows, per file.
3. Document an explicit handling rule and show sensitivity: (a) keep all rows with grouped splitting (current); (b) drop all rows in repeated combinations; (c) keep one row by a stated rule. Report at least (a) and (b).
4. Describe `(SOURCE_FILE, CODE)` grouping as preventing same-course leakage only, not resolving conflicts or proving global identity.
5. Mention that suspended/exempt rows were removed before analysis and that this removal eliminated 115 further raw repeats.
6. Ask the data owner what a repeated student-course-semester under different classes represents.

## Appendix: files

- `full_investigation.py`: reproduces §2, §3 and §6 from the raw and cleaned folders (`python -I full_investigation.py <raw_dir> <cleaned_dir> <out_dir>`; ~15 s).
- `validate_from_outputs.py`: reproduces the pattern/pair statistics of §4, §5, §7 from OpenWork's tables.
- `full_outputs/`: `raw_rows_removed_by_pipeline.csv`, `cleaned_to_raw_row_map.csv`, `duplicate_record_details_with_raw_rows.csv`, `duplicate_traceability.csv`, `raw_cleaned_alignment_by_file.csv`, `raw_level_repeats_by_file.csv`, `full_metrics.json`.
- `outputs/`: `combination_differences_nan_aware.csv`, `representative_examples.csv`, `per_file_counts.csv`, `validation_metrics.json`.
