"""Read-only investigation of repeated student/course/semester observations.

Run from any directory with: py -3 duplicate_investigation/investigation.py
Requires pandas and openpyxl. Writes all generated outputs beside this script.
Never modifies raw/ or cleaned/.
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
RAW, CLEAN = ROOT / "raw", ROOT / "cleaned"
df = pd.read_csv(CLEAN / "ADY201m_combined_cleaned.csv", low_memory=False)
key = ["SOURCE_FILE", "CODE", "Mã học kỳ"]
raw_files = sorted(RAW.glob("*.xlsx"))
course_csvs = sorted(p for p in CLEAN.glob("*_cleaned.csv") if p.name != "ADY201m_combined_cleaned.csv")
sizes = df.groupby(key, dropna=False).size().rename("n_occurrences").reset_index()
dups = sizes[sizes.n_occurrences > 1].sort_values(key)
members = df.merge(dups, on=key, how="inner", validate="many_to_one")
score_cols = [c for c in df.columns if re.search(
    r"(LABS|ASSIGNMENTS|FINAL EXAM|TOTAL|PROGRESS TESTS|EXERCISES|QUIZ|PRACTICAL EXAM|PRESENTATION|ACTIVE LEARNING|PROJECT|WORKSHOP)", c, re.I)]

groups = []
for _, g in members.groupby(key, dropna=False, sort=False):
    g = g.sort_values("ROW_IDX_ORIG", kind="stable")
    first = g.iloc[0]
    distinct = {c: g[c].nunique(dropna=True) for c in df.columns if c not in key + ["ROW_IDX_ORIG"]}
    diff_scores = [c for c in score_cols if distinct.get(c, 0) > 1]
    n_class = distinct.get("Mã lớp", 0)
    n_status, n_target = distinct.get("STATUS_CLEAN", 0), distinct.get("TARGET", 0)
    n_total = distinct.get("TOTAL", 0)
    exact = bool(g.drop(columns=["ROW_IDX_ORIG"]).duplicated(keep=False).all())
    outcome_conflict = n_status > 1 or n_target > 1
    assessment_conflict = bool(diff_scores) or n_total > 1
    if exact:
        pattern = "Exact duplicate records (apart from traceability index)"
    elif outcome_conflict and assessment_conflict:
        pattern = "Conflicting assessment records and outcome labels"
    elif outcome_conflict:
        pattern = "Conflicting outcome labels"
    elif assessment_conflict:
        pattern = "Conflicting assessment records"
    elif n_class > 1:
        pattern = "Different class labels; scores/outcomes otherwise identical"
    else:
        pattern = "Same scores/outcomes; other-field differences or unresolved administrative distinction"
    rec = {c: first[c] for c in key}
    rec.update(
        n_occurrences=len(g), row_indices=";".join(str(int(x)) for x in g.ROW_IDX_ORIG),
        csv_data_rows=";".join(str(int(x)+2) for x in g.ROW_IDX_ORIG),
        n_distinct_classes=n_class, classes=" || ".join(sorted(g["Mã lớp"].dropna().astype(str).unique())),
        n_distinct_STATUS_CLEAN=n_status, statuses=" || ".join(sorted(g.STATUS_CLEAN.dropna().astype(str).unique())),
        n_distinct_TARGET=n_target, targets=" || ".join(sorted(g.TARGET.dropna().astype(str).unique())),
        n_distinct_TOTAL=n_total,
        n_distinct_FINAL_EXAM=sum(g[c].nunique(dropna=True) > 1 for c in df.columns if c.startswith("FINAL EXAM")),
        n_score_columns_differing=len(diff_scores), differing_score_columns=";".join(diff_scores),
        exact_duplicate_all_but_row_idx=exact,
        n_semesters_for_code_in_file=df[(df.SOURCE_FILE == first.SOURCE_FILE) & (df.CODE == first.CODE)]["Mã học kỳ"].nunique(),
        n_files_for_code_all=df[df.CODE == first.CODE].SOURCE_FILE.nunique(),
        n_semesters_for_code_all=df[df.CODE == first.CODE]["Mã học kỳ"].nunique(), pattern=pattern)
    groups.append(rec)
group_detail = pd.DataFrame(groups)
group_detail.to_csv(OUT / "duplicate_combinations.csv", index=False, encoding="utf-8-sig")
detail = members.merge(group_detail[key + ["pattern"]], on=key, validate="many_to_one")
detail["CSV_DATA_ROW"] = detail.ROW_IDX_ORIG + 2  # header is Excel/CSV line 1

    # Compare every duplicated cleaned record to its raw workbook by recorded row index,
    # then independently by (semester, CODE, class, NO.) to detect index alignment errors.
raw_map = {p.name: p for p in raw_files}
idx_checks, key_checks = [], []
for source_name, g in detail.groupby("SOURCE_FILE", sort=True):
    raw_path = raw_map.get(source_name)
    if raw_path is None:
        continue
    raw = pd.read_excel(raw_path)
    overlapping = [c for c in raw.columns if c in df.columns]
    indexed = g.set_index("ROW_IDX_ORIG", drop=False)
    for ix in sorted(set(int(v) for v in g.ROW_IDX_ORIG)):
        r = indexed.loc[ix]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        idx_rec = {"SOURCE_FILE": source_name, "CODE": r.CODE, "Mã học kỳ": r["Mã học kỳ"],
                   "ROW_IDX_ORIG": ix, "RAW_DATAFRAME_INDEX": ix,
                   "RAW_EXCEL_ROW_APPROX": ix + 2, "CLEANED_CSV_DATA_ROW": ix + 2,
                   "pattern": r.pattern}
        if ix >= len(raw):
            idx_rec.update(match_status="ROW_INDEX_OUT_OF_RANGE", n_mismatches=np.nan, mismatch_columns="")
        else:
            rr = raw.iloc[ix]
            mismatches = []
            for c in overlapping:
                a, b = rr[c], r[c]
                if pd.isna(a) and pd.isna(b): continue
                if pd.isna(a) != pd.isna(b): mismatches.append(c); continue
                if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                    if not np.isclose(float(a), float(b), equal_nan=True): mismatches.append(c)
                elif str(a).strip() != str(b).strip(): mismatches.append(c)
            idx_rec.update(match_status="ROW_INDEX_AND_VALUES_MATCH" if not mismatches else "ROW_INDEX_MATCH_VALUES_DIFFER",
                           n_mismatches=len(mismatches), mismatch_columns=";".join(mismatches))
        idx_checks.append(idx_rec)

        candidates = raw[(raw["Mã học kỳ"].astype(str) == str(r["Mã học kỳ"])) & (raw.CODE.astype(str) == str(r.CODE))]
        if "Mã lớp" in raw and pd.notna(r["Mã lớp"]): candidates = candidates[candidates["Mã lớp"].astype(str) == str(r["Mã lớp"])]
        if "NO." in raw and pd.notna(r["NO."]): candidates = candidates[candidates["NO."].astype(str) == str(r["NO."])]
        key_rec = {"SOURCE_FILE": source_name, "CODE": r.CODE, "Mã học kỳ": r["Mã học kỳ"],
                   "Mã lớp": r["Mã lớp"], "NO.": r["NO."], "ROW_IDX_ORIG": ix,
                   "candidate_count": len(candidates), "candidate_raw_indices": ";".join(map(str, candidates.index))}
        key_rec["raw_candidate_records"] = " || ".join(
            json.dumps({c: (None if pd.isna(candidates.iloc[j][c]) else candidates.iloc[j][c])
                        for c in candidates.columns}, ensure_ascii=False, default=str)
            for j in range(len(candidates)))
        if len(candidates) == 1:
            rr = candidates.iloc[0]
            mismatches = []
            for c in overlapping:
                a, b = rr[c], r[c]
                if pd.isna(a) and pd.isna(b): continue
                if pd.isna(a) != pd.isna(b): mismatches.append(c); continue
                if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                    if not np.isclose(float(a), float(b), equal_nan=True): mismatches.append(c)
                elif str(a).strip() != str(b).strip(): mismatches.append(c)
            key_rec.update(key_match_status="UNIQUE_RAW_MATCH_ALL_VALUES_EQUAL" if not mismatches else "UNIQUE_RAW_MATCH_WITH_VALUE_MISMATCHES",
                           raw_match_index=int(candidates.index[0]), mismatch_columns=";".join(mismatches))
        elif len(candidates) == 0:
            key_rec.update(key_match_status="NO_RAW_MATCH", raw_match_index=np.nan, mismatch_columns="")
        else:
            key_rec.update(key_match_status="AMBIGUOUS_RAW_MATCH", raw_match_index=np.nan, mismatch_columns="")
        key_checks.append(key_rec)

pd.DataFrame(idx_checks).to_csv(OUT / "source_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(key_checks).to_csv(OUT / "source_key_match.csv", index=False, encoding="utf-8-sig")

# Confirm the course CSVs mirror the raw files' row-index accounting and dataset provenance.
file_checks = []
for p in course_csvs:
    cdf = pd.read_csv(p, low_memory=False)
    source_name = cdf.SOURCE_FILE.dropna().iloc[0] if "SOURCE_FILE" in cdf and cdf.SOURCE_FILE.notna().any() else ""
    valid = {"SOURCE_FILE", "ROW_IDX_ORIG"} <= set(cdf.columns)
    file_checks.append({"cleaned_file": p.name, "rows": len(cdf), "source_file_value": source_name,
                        "raw_file_found": source_name in raw_map,
                        "row_idx_unique": bool(not cdf.duplicated(["SOURCE_FILE", "ROW_IDX_ORIG"]).any()) if valid else False,
                        "row_idx_range_valid": bool(cdf.ROW_IDX_ORIG.min() == 0 and cdf.ROW_IDX_ORIG.max() == len(cdf)-1) if valid and len(cdf) else False})
pd.DataFrame(file_checks).to_csv(OUT / "file_provenance_checks.csv", index=False, encoding="utf-8-sig")

# Raw-vs-course-cleaned reconciliation at file level. The current raw/ and cleaned/
# directories are not necessarily the same snapshot used by the pipeline metadata.
reconciliation = []
for p in course_csvs:
    cdf = pd.read_csv(p, low_memory=False)
    src = cdf.SOURCE_FILE.dropna().iloc[0]
    raw = pd.read_excel(raw_map[src])
    overlapping = [c for c in raw.columns if c in cdf.columns]
    ixmatch = 0
    for _, r in cdf.iterrows():
        ix = int(r.ROW_IDX_ORIG)
        if ix < 0 or ix >= len(raw): continue
        rr = raw.iloc[ix]
        if all((pd.isna(rr[c]) and pd.isna(r[c])) or
               (pd.notna(rr[c]) and pd.notna(r[c]) and
                ((np.isclose(float(rr[c]), float(r[c])) if isinstance(rr[c], (int,float,np.number)) and isinstance(r[c], (int,float,np.number)) else str(rr[c]).strip() == str(r[c]).strip())))
               for c in overlapping):
            ixmatch += 1
    reconciliation.append({"SOURCE_FILE": src, "raw_rows": len(raw), "cleaned_course_rows": len(cdf),
                           "row_count_difference_raw_minus_cleaned": len(raw)-len(cdf),
                           "cleaned_rows_matching_raw_at_ROW_IDX_ORIG": ixmatch,
                           "overlapping_columns_checked": ";".join(overlapping)})
pd.DataFrame(reconciliation).to_csv(OUT / "raw_cleaned_reconciliation.csv", index=False, encoding="utf-8-sig")

same_code = df.groupby(["SOURCE_FILE", "CODE"], dropna=False).agg(
    n_rows=("CODE", "size"), n_semesters=("Mã học kỳ", "nunique"),
    semesters=("Mã học kỳ", lambda s: ";".join(sorted(s.dropna().astype(str).unique())))).reset_index()
same_code.to_csv(OUT / "code_semester_patterns.csv", index=False, encoding="utf-8-sig")
cross = df.groupby("CODE", dropna=False).agg(
    n_rows=("CODE", "size"), n_files=("SOURCE_FILE", "nunique"),
    files=("SOURCE_FILE", lambda s: ";".join(sorted(s.dropna().astype(str).unique()))),
    n_semesters=("Mã học kỳ", "nunique"), semesters=("Mã học kỳ", lambda s: ";".join(sorted(s.dropna().astype(str).unique())))).reset_index()
cross[cross.n_files > 1].to_csv(OUT / "cross_course_code_patterns.csv", index=False, encoding="utf-8-sig")
pattern_summary = group_detail.groupby("pattern").agg(
    n_combinations=("CODE", "size"), n_rows=("n_occurrences", "sum"),
    excess_rows=("n_occurrences", lambda s: int((s-1).sum()))).reset_index()
pattern_summary.to_csv(OUT / "pattern_summary.csv", index=False, encoding="utf-8-sig")
reps = pd.concat([g.head(3) for _, g in group_detail.groupby("pattern", sort=True)], ignore_index=True)
reps.to_csv(OUT / "representative_patterns.csv", index=False, encoding="utf-8-sig")

idx_df, key_df = pd.DataFrame(idx_checks), pd.DataFrame(key_checks)
metrics = {"combined_rows": len(df), "combined_columns": len(df.columns), "raw_xlsx_count": len(raw_files),
           "course_cleaned_csv_count": len(course_csvs), "repeated_combinations": len(group_detail),
           "rows_in_repeated_combinations": int(group_detail.n_occurrences.sum()),
           "excess_rows": int((group_detail.n_occurrences-1).sum()),
           "frequency_distribution": {str(int(k)): int(v) for k, v in group_detail.n_occurrences.value_counts().sort_index().items()},
           "pattern_counts": {r.pattern: int(r.n_combinations) for r in pattern_summary.itertuples()},
           "index_source_status_counts": {str(k): int(v) for k, v in idx_df.match_status.value_counts().items()},
           "key_source_status_counts": {str(k): int(v) for k, v in key_df.key_match_status.value_counts().items()},
           "raw_row_index_checks": len(idx_df), "raw_key_match_checks": len(key_df)}
(OUT / "investigation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
