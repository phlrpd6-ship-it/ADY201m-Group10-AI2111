"""
ADY201m EWS – Data Cleaning Pipeline (v2.0)
=============================================
Reads the 18 partially cleaned _TongHop.xlsx files and produces:
  • per-file cleaned CSVs  → data/cleaned/
  • combined cleaned CSV   → data/cleaned/ADY201m_combined_cleaned.csv
  • modeling-ready CSV     → data/modeling_eligible/ADY201m_modeling_ready.csv
  • exclusions/flags report → reports/exclusions/
  • missingness report      → reports/missingness/
  • schema report           → reports/schema/
  • quality diagnostics     → reports/quality/
  • Q6 verification         → reports/quality/q6_verification.csv

v2.0 changes:
  - FLAG_TIMING_UNCERTAIN removed (PRACTICAL EXAM / PROJECT confirmed pre-final).
  - Q6 zero-component verification added.
  - Modeling-ready dataset produced with explicit feature list.
  - Modeling safety checks integrated.

Usage:
    py src/clean_data.py

Core rules:
  - Source files are NEVER modified.
  - No imputation of assessment scores.
  - No reconstruction / overwriting of TOTAL.
  - TOTAL and FINAL are preserved for reference but must NOT be used as predictors.
  - SOURCE_FILE is added to every row for traceability.
"""
import sys, io, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# ── Import config ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PARTIALLY_CLEANED, CLEANED_DIR, MODELING_ELIGIBLE,
    MISSINGNESS_DIR, SCHEMA_DIR, QUALITY_DIR, EXCLUSIONS_DIR,
    ID_COLS, META_COLS, ADDED_COLS,
    STATUS_TARGET_MAP, STATUS_EXCLUDED, TRUNCATED_STATUS_MAP,
    FINAL_EXAM_PREFIX, PREFINAL_VALID_PREFIXES,
    ML_EXCLUDE_COLS,
    SCORE_MIN, SCORE_MAX, EXPECTED_FILES, FILE_NO_PREDICTORS,
    ensure_dirs,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def score_cols(df):
    """Return assessment-score column names (everything except ID/META/TOTAL/flags)."""
    exclude = set(ID_COLS + ["STATUS", "STATUS_CLEAN", "TARGET", "TOTAL",
                             "SOURCE_FILE", "ROW_IDX_ORIG",
                             "HAS_PREFINAL", "HAS_PREFINAL_EXCL_PE",
                             "NO_ASSESSMENT_DATA", "FLAG_120102BAA",
                             "FLAG_SUSPICIOUS_TOTAL", "FLAG_NO_SCORE_ERA"])
    return [c for c in df.columns if c not in exclude]


def final_exam_cols(df):
    """Return columns that are FINAL EXAM variants."""
    return [c for c in df.columns if c.startswith(FINAL_EXAM_PREFIX)]


def prefinal_cols(df):
    """Return non-FINAL assessment columns (includes PRACTICAL EXAM / PROJECT)."""
    fe = set(final_exam_cols(df))
    return [c for c in score_cols(df) if c not in fe]


def prefinal_excl_pe_cols(df):
    """Return pre-final cols excluding PRACTICAL EXAM and PROJECT."""
    return [c for c in prefinal_cols(df)
            if not any(c.startswith(p) for p in PREFINAL_VALID_PREFIXES)]


def practical_project_cols(df):
    """Return PRACTICAL EXAM and PROJECT columns (confirmed pre-final)."""
    return [c for c in score_cols(df)
            if any(c.startswith(p) for p in PREFINAL_VALID_PREFIXES)]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 – Read & tag each file
# ═══════════════════════════════════════════════════════════════════════════

def load_and_tag(filepath: Path) -> pd.DataFrame:
    """Load a single xlsx, add traceability columns.  Nothing is modified."""
    df = pd.read_excel(filepath)
    df["SOURCE_FILE"] = filepath.name
    df["ROW_IDX_ORIG"] = range(len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 – Validate & verify STATUS mapping (do NOT modify STATUS)
# ═══════════════════════════════════════════════════════════════════════════

def verify_status_mapping(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Verify that STATUS_CLEAN and TARGET are consistent with the established
    mapping.  Return (df, list of flagged rows as dicts).
    
    Does NOT modify STATUS or STATUS_CLEAN – those were set in the partially
    cleaned data.  Only flags inconsistencies.
    """
    flags = []
    for idx, row in df.iterrows():
        status_clean = row["STATUS_CLEAN"]
        target = row["TARGET"]
        
        # Check if STATUS_CLEAN is one of the expected values
        if status_clean in STATUS_TARGET_MAP:
            expected_target = STATUS_TARGET_MAP[status_clean]
            if target != expected_target:
                flags.append({
                    "source_file": row.get("SOURCE_FILE", ""),
                    "row_idx_orig": row.get("ROW_IDX_ORIG", idx),
                    "CODE": row.get("CODE", ""),
                    "STATUS": row.get("STATUS", ""),
                    "STATUS_CLEAN": status_clean,
                    "TARGET": target,
                    "expected_target": expected_target,
                    "reason": f"TARGET mismatch: STATUS_CLEAN='{status_clean}' expects TARGET={expected_target}, got {target}",
                    "remains_in_dataset": True,
                })
        elif status_clean in STATUS_EXCLUDED:
            if not pd.isna(target):
                flags.append({
                    "source_file": row.get("SOURCE_FILE", ""),
                    "row_idx_orig": row.get("ROW_IDX_ORIG", idx),
                    "CODE": row.get("CODE", ""),
                    "STATUS": row.get("STATUS", ""),
                    "STATUS_CLEAN": status_clean,
                    "TARGET": target,
                    "expected_target": "NaN",
                    "reason": f"Excluded status '{status_clean}' should have TARGET=NaN",
                    "remains_in_dataset": True,
                })
        else:
            # Unknown / unmapped status
            flags.append({
                "source_file": row.get("SOURCE_FILE", ""),
                "row_idx_orig": row.get("ROW_IDX_ORIG", idx),
                "CODE": row.get("CODE", ""),
                "STATUS": row.get("STATUS", ""),
                "STATUS_CLEAN": status_clean,
                "TARGET": target,
                "expected_target": "UNKNOWN",
                "reason": f"Unknown STATUS_CLEAN='{status_clean}' — flagged for review",
                "remains_in_dataset": True,
            })
    
    return df, flags


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 – Add diagnostic flags
# ═══════════════════════════════════════════════════════════════════════════

def add_diagnostic_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flag columns for downstream filtering.  No rows removed.
    
    v2.0: FLAG_TIMING_UNCERTAIN removed — PRACTICAL EXAM and PROJECT are
    confirmed as pre-final assessments.
    """
    sc = score_cols(df)
    pf = prefinal_cols(df)
    pf_no_pe = prefinal_excl_pe_cols(df)
    
    # HAS_PREFINAL: at least one non-FINAL assessment score present
    if pf:
        df["HAS_PREFINAL"] = df[pf].notna().any(axis=1)
    else:
        df["HAS_PREFINAL"] = False
    
    # HAS_PREFINAL_EXCL_PE: excluding PRACTICAL EXAM / PROJECT
    if pf_no_pe:
        df["HAS_PREFINAL_EXCL_PE"] = df[pf_no_pe].notna().any(axis=1)
    else:
        df["HAS_PREFINAL_EXCL_PE"] = False
    
    # NO_ASSESSMENT_DATA: no score columns populated at all
    if sc:
        df["NO_ASSESSMENT_DATA"] = df[sc].isna().all(axis=1)
    else:
        df["NO_ASSESSMENT_DATA"] = True
    
    # FLAG_120102BAA: this file has no usable assessment predictors
    df["FLAG_120102BAA"] = df["SOURCE_FILE"] == FILE_NO_PREDICTORS
    
    # FLAG_SUSPICIOUS_TOTAL: 120102BAA TOTAL is all-zeros placeholder
    if df["SOURCE_FILE"].iloc[0] == FILE_NO_PREDICTORS:
        df["FLAG_SUSPICIOUS_TOTAL"] = df["TOTAL"].notna()  # all 388 non-null values are 0.0 placeholders
    else:
        df["FLAG_SUSPICIOUS_TOTAL"] = False
    
    # FLAG_NO_SCORE_ERA: rows from semesters where no score data exists at all
    # (determined per file: semesters where ALL score columns are NaN for ALL rows)
    if sc:
        sem_has_any = df.groupby("Mã học kỳ")[sc].apply(
            lambda g: g.notna().any().any()
        )
        no_score_sems = set(sem_has_any[~sem_has_any].index)
        df["FLAG_NO_SCORE_ERA"] = df["Mã học kỳ"].isin(no_score_sems)
    else:
        df["FLAG_NO_SCORE_ERA"] = True
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 – Generate per-file reports
# ═══════════════════════════════════════════════════════════════════════════

def generate_missingness_report(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Generate missingness report for one file."""
    rows = []
    sc = score_cols(df)
    for c in df.columns:
        n_miss = int(df[c].isna().sum())
        if n_miss == 0:
            continue
        if c in sc:
            # Check structural vs partial pattern
            g = df.groupby("Mã học kỳ")[c].apply(lambda s: s.notna().mean())
            if ((g == 0) | (g == 1)).all():
                pattern = "structural (all-or-nothing per semester)"
            else:
                partial_sems = g[(g > 0) & (g < 1)]
                pattern = f"partial in {len(partial_sems)} semester(s): {dict(partial_sems.round(3))}"
        elif c == "TOTAL":
            pattern = "TOTAL column"
        else:
            pattern = "other"
        rows.append({
            "source_file": filename,
            "column": c,
            "n_missing": n_miss,
            "n_rows": len(df),
            "pct_missing": round(100 * n_miss / len(df), 2),
            "pattern": pattern,
        })
    return pd.DataFrame(rows)


def generate_schema_report(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Generate schema/columns report for one file."""
    sc = score_cols(df)
    fe = final_exam_cols(df)
    pf = prefinal_cols(df)
    pf_no_pe = prefinal_excl_pe_cols(df)
    pp = practical_project_cols(df)
    
    return pd.DataFrame([{
        "source_file": filename,
        "n_rows": len(df),
        "n_semesters": df["Mã học kỳ"].nunique(),
        "n_classes": df.groupby(["Mã học kỳ", "Mã lớp"]).ngroups,
        "score_columns": "; ".join(sc) if sc else "(none)",
        "final_exam_columns": "; ".join(fe) if fe else "(none)",
        "prefinal_columns": "; ".join(pf) if pf else "(none)",
        "prefinal_excl_PE": "; ".join(pf_no_pe) if pf_no_pe else "(none)",
        "practical_project_cols": "; ".join(pp) if pp else "(none)",
        "has_TOTAL": "TOTAL" in df.columns,
        "total_non_null": int(df["TOTAL"].notna().sum()) if "TOTAL" in df.columns else 0,
    }])


def generate_code_repetition_report(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Diagnostics for CODE repetition within this file."""
    rows = []
    # Within-file CODE repetition
    code_counts = df.CODE.value_counts()
    repeated = code_counts[code_counts > 1]
    
    # Within-semester repetition
    sem_code = df.groupby(["Mã học kỳ", "CODE"]).size().reset_index(name="count")
    sem_repeated = sem_code[sem_code["count"] > 1]
    
    rows.append({
        "source_file": filename,
        "total_rows": len(df),
        "distinct_CODEs": df.CODE.nunique(),
        "CODEs_repeated_in_file": len(repeated),
        "rows_with_repeated_CODE_in_file": int(repeated.sum()),
        "CODEs_repeated_same_semester": len(sem_repeated),
        "rows_with_repeated_CODE_same_semester": int(
            df.merge(sem_repeated[["Mã học kỳ", "CODE"]], on=["Mã học kỳ", "CODE"]).shape[0]
        ) if len(sem_repeated) > 0 else 0,
    })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 – Build exclusions/flags report
# ═══════════════════════════════════════════════════════════════════════════

def build_exclusion_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Build a detailed row-level flags report.
    
    Rows are NOT removed — this report documents which rows have issues.
    """
    flagged = []
    
    for idx, row in df.iterrows():
        reasons = []
        
        if row.get("FLAG_120102BAA", False):
            reasons.append("120102BAA: no usable assessment predictors")
        
        if row.get("FLAG_SUSPICIOUS_TOTAL", False):
            reasons.append("TOTAL is a placeholder (0.0 in 120102BAA)")
        
        if row.get("NO_ASSESSMENT_DATA", False) and not row.get("FLAG_120102BAA", False):
            reasons.append("No assessment data (all score columns NaN)")
        
        if row.get("FLAG_NO_SCORE_ERA", False) and not row.get("FLAG_120102BAA", False):
            reasons.append("From a no-score-data schema era (e.g. SP2017/SU2017)")
        
        if not row.get("HAS_PREFINAL", True):
            if not row.get("FLAG_120102BAA", False) and not row.get("NO_ASSESSMENT_DATA", False):
                reasons.append("No pre-final assessment data (only FINAL EXAM present)")
        
        if reasons:
            flagged.append({
                "source_file": row.get("SOURCE_FILE", ""),
                "row_idx_orig": row.get("ROW_IDX_ORIG", idx),
                "CODE": row.get("CODE", ""),
                "semester": row.get("Mã học kỳ", ""),
                "class": row.get("Mã lớp", ""),
                "STATUS": row.get("STATUS", ""),
                "STATUS_CLEAN": row.get("STATUS_CLEAN", ""),
                "TARGET": row.get("TARGET", ""),
                "reasons": " | ".join(reasons),
                "remains_in_cleaned": True,
                "modeling_eligible": False,
            })
    
    return pd.DataFrame(flagged) if flagged else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 – Q6 verification: 799 "Not Passed" with TOTAL≥5 & FINAL≥4
# ═══════════════════════════════════════════════════════════════════════════

def verify_q6_contradictions(combined: pd.DataFrame) -> dict:
    """
    Verify the ~799 rows where STATUS_CLEAN='Not Passed' but TOTAL≥5 and FINAL≥4.
    
    The project rule: a student fails if any assessment component is exactly 0.
    We programmatically check whether each such row has at least one component = 0.
    
    Returns a summary dict and saves detailed CSV.
    """
    # Identify all score columns in the combined dataset
    sc = score_cols(combined)
    
    # Get FINAL EXAM columns to separate them from component columns
    fe_cols = final_exam_cols(combined)
    
    # Component columns = all score columns (pre-final scores that contribute)
    component_cols = [c for c in sc if c not in fe_cols]
    
    # Filter: Not Passed + TOTAL >= 5 + any FINAL EXAM >= 4
    mask_np = combined["STATUS_CLEAN"] == "Not Passed"
    mask_total = combined["TOTAL"] >= 5
    
    # FINAL EXAM >= 4: check if ANY FINAL EXAM column has value >= 4
    if fe_cols:
        mask_final = (combined[fe_cols] >= 4).any(axis=1)
    else:
        mask_final = pd.Series(False, index=combined.index)
    
    q6_mask = mask_np & mask_total & mask_final
    q6_rows = combined.loc[q6_mask].copy()
    n_total = len(q6_rows)
    
    if n_total == 0:
        return {
            "total_contradictions": 0,
            "explained_by_zero": 0,
            "unexplained": 0,
            "pct_explained": 100.0,
            "status": "No contradictions found",
        }
    
    # For each row, check if ANY component column has value exactly 0
    # We check all score columns (including FINAL) to be thorough,
    # but report specifically on component columns
    if component_cols:
        has_zero_component = (q6_rows[component_cols] == 0).any(axis=1)
    else:
        has_zero_component = pd.Series(False, index=q6_rows.index)
    
    # Also check FINAL EXAM columns for zeros
    if fe_cols:
        has_zero_final = (q6_rows[fe_cols] == 0).any(axis=1)
    else:
        has_zero_final = pd.Series(False, index=q6_rows.index)
    
    # A row has at least one zero in any assessment
    has_any_zero = has_zero_component | has_zero_final
    
    n_explained = int(has_any_zero.sum())
    n_unexplained = n_total - n_explained
    pct_explained = round(100 * n_explained / n_total, 2) if n_total > 0 else 0
    
    # Build detail report
    q6_rows["has_zero_component"] = has_zero_component
    q6_rows["has_zero_final"] = has_zero_final
    q6_rows["has_any_zero"] = has_any_zero
    q6_rows["explained_by_zero_rule"] = has_any_zero
    
    # Identify which columns are zero for unexplained rows
    zero_detail = []
    for idx, row in q6_rows.iterrows():
        zero_cols_list = []
        for c in component_cols + fe_cols:
            if pd.notna(row.get(c)) and row[c] == 0:
                zero_cols_list.append(c)
        zero_detail.append("; ".join(zero_cols_list) if zero_cols_list else "(none)")
    q6_rows["zero_columns"] = zero_detail
    
    # Save detailed report
    report_cols = ["SOURCE_FILE", "ROW_IDX_ORIG", "CODE", "Mã học kỳ", "STATUS_CLEAN",
                   "TARGET", "TOTAL"] + fe_cols + component_cols + \
                  ["has_zero_component", "has_zero_final", "has_any_zero",
                   "explained_by_zero_rule", "zero_columns"]
    report_cols = [c for c in report_cols if c in q6_rows.columns]
    q6_report = q6_rows[report_cols]
    q6_report.to_csv(QUALITY_DIR / "q6_verification.csv", index=False, encoding="utf-8-sig")
    
    # If there are unexplained rows, save them separately
    if n_unexplained > 0:
        unexplained = q6_rows[~has_any_zero]
        unexplained[report_cols].to_csv(
            QUALITY_DIR / "q6_unexplained_rows.csv", index=False, encoding="utf-8-sig"
        )
    
    # Determine resolution status
    if n_unexplained == 0:
        status = "RESOLVED: All contradictions explained by the zero-component rule"
    elif pct_explained >= 95:
        status = f"MOSTLY RESOLVED: {pct_explained}% explained; {n_unexplained} rows need review"
    else:
        status = f"PARTIALLY RESOLVED: {pct_explained}% explained; {n_unexplained} rows need review"
    
    return {
        "total_contradictions": n_total,
        "explained_by_zero": n_explained,
        "unexplained": n_unexplained,
        "pct_explained": pct_explained,
        "status": status,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 – Produce modeling-ready dataset
# ═══════════════════════════════════════════════════════════════════════════

def produce_modeling_dataset(combined: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Produce the final modeling-ready dataset.
    
    Inclusion criteria for modeling rows:
      - TARGET is not NaN (i.e., not Suspended/Exempt — already excluded)
      - HAS_PREFINAL = True (has at least one pre-final assessment score)
      - NO_ASSESSMENT_DATA = False
    
    Feature selection:
      - Exclude all FINAL EXAM columns (leakage)
      - Exclude TOTAL (leakage)
      - Exclude CODE, STATUS, STATUS_CLEAN, ID columns (not predictors)
      - Exclude all diagnostic flag columns
      - Keep all pre-final assessment columns (including PRACTICAL EXAM, PROJECT)
      - Keep QUIZ and PROGRESS TEST columns SEPARATE (not merged)
    
    Returns: (modeling_df, feature_info_dict)
    """
    # Filter to modeling-eligible rows
    mask = (
        combined["TARGET"].notna() &
        combined["HAS_PREFINAL"] &
        ~combined["NO_ASSESSMENT_DATA"]
    )
    modeling = combined.loc[mask].copy()
    
    # Determine ML feature columns
    all_cols = list(modeling.columns)
    
    # Build explicit exclusion set
    exclude_set = set(ML_EXCLUDE_COLS)
    
    # Add all FINAL EXAM columns to exclusion
    for c in all_cols:
        if c.startswith(FINAL_EXAM_PREFIX):
            exclude_set.add(c)
    
    # ML features = remaining assessment score columns
    ml_features = [c for c in all_cols if c not in exclude_set]
    
    # Safety checks
    safety = {
        "FINAL_EXAM_excluded": all(c not in ml_features for c in all_cols if c.startswith(FINAL_EXAM_PREFIX)),
        "TOTAL_excluded": "TOTAL" not in ml_features,
        "CODE_excluded": "CODE" not in ml_features,
        "TARGET_excluded": "TARGET" not in ml_features,
        "STATUS_excluded": "STATUS" not in ml_features and "STATUS_CLEAN" not in ml_features,
        "SOURCE_FILE_excluded": "SOURCE_FILE" not in ml_features,
        "PRACTICAL_EXAM_included": any(c.startswith("PRACTICAL EXAM") for c in ml_features),
        "PROJECT_included": any(c.startswith("PROJECT") for c in ml_features),
        "QUIZ_PROGRESS_separate": not any("QUIZ_PROGRESS" in c for c in ml_features),
        "no_flag_cols_in_features": not any(c.startswith("FLAG_") or c.startswith("HAS_") or c == "NO_ASSESSMENT_DATA" for c in ml_features),
        "raw_scores_preserved": True,  # no normalization/reweighting applied
    }
    
    all_safe = all(safety.values())
    
    # Categorize columns
    metadata_cols = [c for c in all_cols if c in exclude_set and c not in
                     [cc for cc in all_cols if cc.startswith(FINAL_EXAM_PREFIX)] and
                     c != "TOTAL"]
    reference_cols = [c for c in all_cols if c.startswith(FINAL_EXAM_PREFIX) or c == "TOTAL"]
    
    feature_info = {
        "ml_features": ml_features,
        "n_ml_features": len(ml_features),
        "target_col": "TARGET",
        "metadata_cols": sorted(set(all_cols) - set(ml_features) - {"TARGET"} - set(reference_cols)),
        "reference_cols": reference_cols,
        "excluded_from_ml": sorted(exclude_set),
        "safety_checks": safety,
        "all_safety_passed": all_safe,
        "n_modeling_rows": len(modeling),
        "target_distribution": {
            "TARGET_0": int((modeling.TARGET == 0).sum()),
            "TARGET_1": int((modeling.TARGET == 1).sum()),
        },
    }
    
    return modeling, feature_info


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    start_time = datetime.now()
    print(f"{'='*70}")
    print(f"ADY201m EWS – Data Cleaning Pipeline v2.0")
    print(f"Started: {start_time.isoformat()}")
    print(f"{'='*70}")
    
    ensure_dirs()
    
    # ── Check input files ────────────────────────────────────────────────
    missing_files = [f for f in EXPECTED_FILES if not (PARTIALLY_CLEANED / f).exists()]
    if missing_files:
        print(f"\nERROR: Missing expected files: {missing_files}")
        sys.exit(1)
    
    found_files = sorted(PARTIALLY_CLEANED.glob("*_TongHop.xlsx"))
    found_names = [f.name for f in found_files]
    extra_files = set(found_names) - set(EXPECTED_FILES)
    if extra_files:
        print(f"\nWARNING: Unexpected files found: {extra_files}")
    
    print(f"\nInput: {len(found_files)} files from {PARTIALLY_CLEANED}")
    
    # ── Process each file ────────────────────────────────────────────────
    all_cleaned = []
    all_missingness = []
    all_schema = []
    all_code_rep = []
    all_status_flags = []
    all_exclusion_flags = []
    file_summaries = []
    
    for filepath in found_files:
        fname = filepath.name
        short = fname.split("_")[0]
        print(f"\n  Processing {fname} ...", end=" ")
        
        # Step 1: Load & tag
        df = load_and_tag(filepath)
        n_orig = len(df)
        
        # Step 2: Verify STATUS mapping
        df, status_flags = verify_status_mapping(df)
        all_status_flags.extend(status_flags)
        
        # Step 3: Add diagnostic flags
        df = add_diagnostic_flags(df)
        
        # Step 4: Generate reports
        miss_report = generate_missingness_report(df, fname)
        schema_report = generate_schema_report(df, fname)
        code_rep = generate_code_repetition_report(df, fname)
        excl_flags = build_exclusion_flags(df)
        
        all_missingness.append(miss_report)
        all_schema.append(schema_report)
        all_code_rep.append(code_rep)
        if len(excl_flags) > 0:
            all_exclusion_flags.append(excl_flags)
        
        # Step 5: Save per-file cleaned CSV (no rows removed, flags added)
        out_path = CLEANED_DIR / f"{short}_cleaned.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        
        all_cleaned.append(df)
        
        sc = score_cols(df)
        pf = prefinal_cols(df)
        
        file_summaries.append({
            "file": fname,
            "rows": n_orig,
            "score_columns": len(sc),
            "prefinal_columns": len(pf),
            "has_prefinal_rows": int(df["HAS_PREFINAL"].sum()),
            "no_assessment_rows": int(df["NO_ASSESSMENT_DATA"].sum()),
            "target_0": int((df.TARGET == 0).sum()),
            "target_1": int((df.TARGET == 1).sum()),
            "status_flags": len(status_flags),
            "exclusion_flags": len(excl_flags),
        })
        
        print(f"OK ({n_orig} rows, {len(sc)} score cols, {len(status_flags)} status flags)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMBINE
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("Combining all files...")
    
    combined = pd.concat(all_cleaned, ignore_index=True)
    print(f"  Combined: {len(combined)} rows from {len(all_cleaned)} files")
    print(f"  TARGET distribution: 0={int((combined.TARGET==0).sum())}, 1={int((combined.TARGET==1).sum())}")
    print(f"  Distinct CODEs: {combined.CODE.nunique()}")
    
    # Save combined
    combined_path = CLEANED_DIR / "ADY201m_combined_cleaned.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {combined_path}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Q6 VERIFICATION
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("Q6 Verification: Not Passed with TOTAL≥5 and FINAL≥4...")
    
    q6_result = verify_q6_contradictions(combined)
    print(f"  Total contradictions found: {q6_result['total_contradictions']}")
    print(f"  Explained by zero-component rule: {q6_result['explained_by_zero']}")
    print(f"  Unexplained: {q6_result['unexplained']}")
    print(f"  Percentage explained: {q6_result['pct_explained']}%")
    print(f"  Status: {q6_result['status']}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE REPORTS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("Generating reports...")
    
    # Missingness
    miss_combined = pd.concat(all_missingness, ignore_index=True)
    miss_combined.to_csv(MISSINGNESS_DIR / "missingness_report.csv", index=False, encoding="utf-8-sig")
    print(f"  Missingness: {len(miss_combined)} entries → {MISSINGNESS_DIR}")
    
    # Schema
    schema_combined = pd.concat(all_schema, ignore_index=True)
    schema_combined.to_csv(SCHEMA_DIR / "schema_report.csv", index=False, encoding="utf-8-sig")
    print(f"  Schema: {len(schema_combined)} entries → {SCHEMA_DIR}")
    
    # CODE repetition
    code_rep_combined = pd.concat(all_code_rep, ignore_index=True)
    code_rep_combined.to_csv(QUALITY_DIR / "code_repetition_report.csv", index=False, encoding="utf-8-sig")
    print(f"  CODE repetition: {len(code_rep_combined)} entries → {QUALITY_DIR}")
    
    # Cross-file CODE analysis
    cross_code = combined.groupby("CODE").agg(
        n_rows=("CODE", "size"),
        n_files=("SOURCE_FILE", "nunique"),
        files=("SOURCE_FILE", lambda x: "; ".join(sorted(x.unique()))),
        n_semesters=("Mã học kỳ", "nunique"),
    ).reset_index()
    cross_code_repeated = cross_code[cross_code.n_rows > 1]
    cross_code_repeated.to_csv(QUALITY_DIR / "code_cross_file_repetition.csv", index=False, encoding="utf-8-sig")
    print(f"  Cross-file CODE repetition: {len(cross_code_repeated)} repeated CODEs → {QUALITY_DIR}")
    
    # Within-semester CODE repetition detail
    sem_code_detail = combined.groupby(["SOURCE_FILE", "Mã học kỳ", "CODE"]).agg(
        n_occurrences=("CODE", "size"),
        classes=("Mã lớp", lambda x: "; ".join(sorted(x.unique()))),
        targets=("TARGET", lambda x: list(x)),
    ).reset_index()
    sem_repeated = sem_code_detail[sem_code_detail.n_occurrences > 1]
    sem_repeated.to_csv(QUALITY_DIR / "code_within_semester_repetition.csv", index=False, encoding="utf-8-sig")
    print(f"  Within-semester CODE repetition: {len(sem_repeated)} groups → {QUALITY_DIR}")
    
    # Status flags
    if all_status_flags:
        status_df = pd.DataFrame(all_status_flags)
        status_df.to_csv(EXCLUSIONS_DIR / "status_mapping_flags.csv", index=False, encoding="utf-8-sig")
        print(f"  Status mapping flags: {len(status_df)} rows → {EXCLUSIONS_DIR}")
    else:
        print("  Status mapping: ✓ All consistent (0 flags)")
    
    # Exclusion flags
    if all_exclusion_flags:
        excl_df = pd.concat(all_exclusion_flags, ignore_index=True)
        excl_df.to_csv(EXCLUSIONS_DIR / "exclusion_flags.csv", index=False, encoding="utf-8-sig")
        print(f"  Exclusion flags: {len(excl_df)} rows → {EXCLUSIONS_DIR}")
    else:
        print("  Exclusion flags: 0 rows")
    
    # File summaries
    summary_df = pd.DataFrame(file_summaries)
    summary_df.to_csv(QUALITY_DIR / "file_processing_summary.csv", index=False, encoding="utf-8-sig")
    
    # ═══════════════════════════════════════════════════════════════════════
    # QUALITY DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("Quality diagnostics...")
    
    # TOTAL diagnostic (from audit S1, S3, S4)
    quality_rows = []
    for df_file in all_cleaned:
        fname = df_file["SOURCE_FILE"].iloc[0]
        sc = score_cols(df_file)
        if sc:
            # S3: TOTAL > 0 but all recorded (non-null) scores are 0.0
            # Among populated score cells, every one is zero
            has_any_score = df_file[sc].notna().any(axis=1)
            # For each row: check that no score column has a value > 0
            any_score_positive = (df_file[sc] > 0).any(axis=1)
            all_recorded_zero = has_any_score & ~any_score_positive
            total_pos = df_file["TOTAL"] > 0
            s3_count = int((all_recorded_zero & total_pos).sum())
            
            # S4: TOTAL = 0 but at least one component > 0
            comp_pos = (df_file[sc] > 0).any(axis=1)
            total_zero = df_file["TOTAL"] == 0
            s4_count = int((total_zero & comp_pos).sum())
        else:
            s3_count = 0
            s4_count = 0
        
        quality_rows.append({
            "source_file": fname,
            "S3_total_pos_scores_zero": s3_count,
            "S4_total_zero_comp_pos": s4_count,
        })
    
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(QUALITY_DIR / "total_anomaly_diagnostic.csv", index=False, encoding="utf-8-sig")
    print(f"  TOTAL anomalies: S3={int(quality_df.S3_total_pos_scores_zero.sum())}, "
          f"S4={int(quality_df.S4_total_zero_comp_pos.sum())}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODELING-READY DATASET
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("Producing modeling-ready dataset...")
    
    modeling_df, feature_info = produce_modeling_dataset(combined)
    
    # Save modeling-ready dataset (full — includes metadata for traceability)
    modeling_path = MODELING_ELIGIBLE / "ADY201m_modeling_ready.csv"
    modeling_df.to_csv(modeling_path, index=False, encoding="utf-8-sig")
    print(f"  Modeling-ready: {len(modeling_df)} rows → {modeling_path}")
    print(f"  TARGET: 0={feature_info['target_distribution']['TARGET_0']}, "
          f"1={feature_info['target_distribution']['TARGET_1']}")
    print(f"  ML features ({feature_info['n_ml_features']}): {feature_info['ml_features']}")
    
    # Safety check report
    print(f"\n  Modeling safety checks:")
    for check, passed in feature_info["safety_checks"].items():
        symbol = "✓" if passed else "✗ FAIL"
        print(f"    [{symbol}] {check}")
    
    if not feature_info["all_safety_passed"]:
        print("\n  *** WARNING: Not all safety checks passed! Review before modeling. ***")
    
    # Save feature info
    with open(MODELING_ELIGIBLE / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)
    
    # Save feature list as CSV for easy reference
    feature_list = []
    for c in feature_info["ml_features"]:
        feature_list.append({
            "column": c,
            "role": "ML_PREDICTOR",
            "type": "pre-final assessment score",
        })
    feature_list.append({
        "column": "TARGET",
        "role": "TARGET",
        "type": "binary outcome (0=Passed, 1=Not Passed/Attendance Fail)",
    })
    for c in feature_info.get("reference_cols", []):
        feature_list.append({
            "column": c,
            "role": "REFERENCE",
            "type": "excluded from ML (leakage/post-outcome)",
        })
    for c in feature_info.get("metadata_cols", []):
        role = "METADATA"
        if c == "CODE":
            type_desc = "anonymized student ID (traceability only)"
        elif c == "SOURCE_FILE":
            type_desc = "course/file identity (stratification only)"
        elif c in ("Mã học kỳ", "Mã lớp", "NO."):
            type_desc = "administrative identifier"
        elif c in ("STATUS", "STATUS_CLEAN"):
            type_desc = "target-derived (excluded)"
        elif c.startswith("FLAG_") or c.startswith("HAS_") or c == "NO_ASSESSMENT_DATA":
            type_desc = "diagnostic flag"
        elif c == "ROW_IDX_ORIG":
            type_desc = "traceability index"
        else:
            type_desc = "metadata"
        feature_list.append({
            "column": c,
            "role": role,
            "type": type_desc,
        })
    
    pd.DataFrame(feature_list).to_csv(
        MODELING_ELIGIBLE / "feature_list.csv", index=False, encoding="utf-8-sig"
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # PIPELINE METADATA
    # ═══════════════════════════════════════════════════════════════════════
    end_time = datetime.now()
    metadata = {
        "pipeline": "ADY201m EWS Data Cleaning",
        "version": "2.0.0",
        "run_timestamp": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "input_dir": str(PARTIALLY_CLEANED),
        "output_dir": str(CLEANED_DIR),
        "files_processed": len(found_files),
        "total_rows": len(combined),
        "target_0": int((combined.TARGET == 0).sum()),
        "target_1": int((combined.TARGET == 1).sum()),
        "distinct_CODEs": int(combined.CODE.nunique()),
        "status_mapping_flags": len(all_status_flags),
        "exclusion_flags": sum(len(ef) for ef in all_exclusion_flags),
        "modeling_ready_rows": len(modeling_df),
        "n_ml_features": feature_info["n_ml_features"],
        "ml_features": feature_info["ml_features"],
        "q6_verification": q6_result,
    }
    
    with open(CLEANED_DIR / "pipeline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"Pipeline completed in {metadata['duration_seconds']:.1f}s")
    print(f"  {metadata['files_processed']} files → {metadata['total_rows']} rows")
    print(f"  Modeling-ready: {metadata['modeling_ready_rows']} rows, {metadata['n_ml_features']} ML features")
    print(f"  Output: {CLEANED_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
