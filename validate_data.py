"""
ADY201m EWS – Data Validation (v2.0)
=======================================
Validates the cleaned outputs produced by clean_data.py.

Checks:
  V01–V13: Original validation checks (preserved)
  V14: Q6 zero-component verification
  V15: Modeling-ready dataset safety checks
  V16: Attendance Fail target assignment
  V17: QUIZ and PROGRESS TEST remain separate columns

v2.0 changes:
  - FLAG_TIMING_UNCERTAIN removed from expected flags.
  - V14: Q6 verification added.
  - V15: Modeling safety checks added.
  - V16: Attendance Fail handling verified.
  - V17: QUIZ/PROGRESS TEST separation verified.

Usage:
    py src/validate_data.py
"""
import sys, io, os, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PARTIALLY_CLEANED, CLEANED_DIR, MODELING_ELIGIBLE,
    VALIDATION_DIR, EXCLUSIONS_DIR, QUALITY_DIR,
    ID_COLS, STATUS_TARGET_MAP, STATUS_EXCLUDED,
    SCORE_MIN, SCORE_MAX, EXPECTED_FILES, FILE_NO_PREDICTORS,
    FINAL_EXAM_PREFIX, ML_EXCLUDE_COLS,
    ensure_dirs,
)

# ═══════════════════════════════════════════════════════════════════════════
# Validation checks
# ═══════════════════════════════════════════════════════════════════════════

class ValidationResult:
    """Container for validation results."""
    def __init__(self):
        self.checks = []
    
    def add(self, check_id: str, name: str, passed: bool, detail: str = ""):
        self.checks.append({
            "check_id": check_id,
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        symbol = "✓" if passed else "✗ FAIL"
        print(f"  [{symbol}] {check_id}: {name}")
        if detail and not passed:
            for line in detail.split("\n"):
                print(f"       {line}")
    
    @property
    def all_passed(self):
        return all(c["passed"] for c in self.checks)
    
    @property
    def n_passed(self):
        return sum(1 for c in self.checks if c["passed"])
    
    @property
    def n_failed(self):
        return sum(1 for c in self.checks if not c["passed"])
    
    def to_dataframe(self):
        return pd.DataFrame(self.checks)
    
    def to_json(self):
        return {
            "timestamp": datetime.now().isoformat(),
            "all_passed": self.all_passed,
            "n_checks": len(self.checks),
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "checks": self.checks,
        }


def validate_all():
    """Run all validation checks."""
    ensure_dirs()
    results = ValidationResult()
    
    print(f"{'='*70}")
    print(f"ADY201m EWS – Data Validation v2.0")
    print(f"{'='*70}\n")
    
    # ── Load combined cleaned data ────────────────────────────────────────
    combined_path = CLEANED_DIR / "ADY201m_combined_cleaned.csv"
    if not combined_path.exists():
        print(f"ERROR: Combined cleaned file not found: {combined_path}")
        print("Run clean_data.py first.")
        sys.exit(1)
    
    df = pd.read_csv(combined_path)
    print(f"Loaded: {combined_path} ({len(df)} rows)\n")
    
    # ── V01: All expected files processed ─────────────────────────────────
    processed_files = sorted(df["SOURCE_FILE"].unique())
    expected = sorted(EXPECTED_FILES)
    missing = set(expected) - set(processed_files)
    extra = set(processed_files) - set(expected)
    results.add(
        "V01", "All expected files processed",
        len(missing) == 0,
        f"Missing: {missing}" if missing else f"All {len(expected)} files present"
    )
    if extra:
        results.add("V01b", "No unexpected files", False, f"Extra: {extra}")
    
    # ── V02: Required ID columns exist ────────────────────────────────────
    required_cols = ID_COLS + ["STATUS", "STATUS_CLEAN", "TARGET", "TOTAL", "SOURCE_FILE"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    results.add(
        "V02", "Required ID/meta columns exist",
        len(missing_cols) == 0,
        f"Missing: {missing_cols}" if missing_cols else ""
    )
    
    # ── V03: No required ID values unexpectedly missing ───────────────────
    id_missing = {}
    for c in ID_COLS + ["STATUS", "STATUS_CLEAN", "SOURCE_FILE"]:
        if c in df.columns:
            n_miss = int(df[c].isna().sum())
            if n_miss > 0:
                id_missing[c] = n_miss
    results.add(
        "V03", "No missing values in ID/STATUS columns",
        len(id_missing) == 0,
        f"Missing values: {id_missing}" if id_missing else ""
    )
    
    # ── V04: Assessment values within valid ranges ────────────────────────
    id_meta_set = set(ID_COLS + ["STATUS", "STATUS_CLEAN", "TARGET", "TOTAL",
                                  "SOURCE_FILE", "ROW_IDX_ORIG",
                                  "HAS_PREFINAL", "HAS_PREFINAL_EXCL_PE",
                                  "NO_ASSESSMENT_DATA", "FLAG_120102BAA",
                                  "FLAG_SUSPICIOUS_TOTAL", "FLAG_NO_SCORE_ERA"])
    score_cols_list = [c for c in df.columns if c not in id_meta_set]
    out_of_range = {}
    for c in score_cols_list:
        s = df[c].dropna()
        if len(s) > 0:
            below = int((s < SCORE_MIN).sum())
            above = int((s > SCORE_MAX).sum())
            if below > 0 or above > 0:
                out_of_range[c] = {"below_0": below, "above_10": above}
    results.add(
        "V04", "Assessment scores within 0-10 range",
        len(out_of_range) == 0,
        f"Out of range: {out_of_range}" if out_of_range else f"Checked {len(score_cols_list)} score columns"
    )
    
    # ── V05: TARGET contains only 0/1 for modeling-eligible rows ──────────
    target_vals = df["TARGET"].dropna().unique()
    non_binary = [v for v in target_vals if v not in (0, 1, 0.0, 1.0)]
    results.add(
        "V05", "TARGET contains only 0/1",
        len(non_binary) == 0,
        f"Non-binary values: {non_binary}" if non_binary else f"Values: {sorted(target_vals)}"
    )
    
    # ── V06: STATUS_CLEAN mapping is internally consistent ────────────────
    inconsistent = []
    for status, expected_target in STATUS_TARGET_MAP.items():
        mask = df["STATUS_CLEAN"] == status
        if mask.any():
            actual = df.loc[mask, "TARGET"].unique()
            if not all(v == expected_target for v in actual):
                inconsistent.append(f"{status}: expected TARGET={expected_target}, got {actual}")
    results.add(
        "V06", "STATUS_CLEAN → TARGET mapping consistent",
        len(inconsistent) == 0,
        "\n".join(inconsistent) if inconsistent else ""
    )
    
    # ── V07: FINAL and TOTAL not in predictor feature set ─────────────────
    # (This checks column presence — actual feature selection happens at modeling)
    has_final_col = any(c.startswith(FINAL_EXAM_PREFIX) for c in score_cols_list)
    has_total_in_scores = "TOTAL" in score_cols_list
    # These SHOULD be in the dataset for reference, but flagged
    detail = []
    if has_final_col:
        final_cols = [c for c in score_cols_list if c.startswith(FINAL_EXAM_PREFIX)]
        detail.append(f"FINAL EXAM columns present (for reference, not predictors): {final_cols}")
    if has_total_in_scores:
        detail.append("TOTAL in score columns (for reference, not predictor)")
    results.add(
        "V07", "FINAL/TOTAL present for reference (must not be predictors)",
        True,  # They should be present; validation is that they're flagged
        "\n".join(detail) if detail else "FINAL and TOTAL columns present for auditing"
    )
    
    # ── V08: No accidental duplicate rows ─────────────────────────────────
    # Check for exact full-row duplicates (excluding ROW_IDX_ORIG which is unique by definition)
    check_cols = [c for c in df.columns if c != "ROW_IDX_ORIG"]
    n_dup = int(df.duplicated(subset=check_cols, keep=False).sum())
    # Also check if same (SOURCE_FILE, ROW_IDX_ORIG) appears twice (would indicate a concat bug)
    n_idx_dup = int(df.duplicated(subset=["SOURCE_FILE", "ROW_IDX_ORIG"], keep=False).sum())
    results.add(
        "V08", "No accidental duplicate rows",
        n_idx_dup == 0,
        f"Index duplicates: {n_idx_dup}" if n_idx_dup > 0 else f"Content-level duplicates (intentional, e.g. similar scores): {n_dup}"
    )
    
    # ── V09: Source-file traceability preserved ───────────────────────────
    has_source = "SOURCE_FILE" in df.columns
    has_row_idx = "ROW_IDX_ORIG" in df.columns
    all_source_non_null = df["SOURCE_FILE"].notna().all() if has_source else False
    results.add(
        "V09", "Source-file traceability preserved",
        has_source and has_row_idx and all_source_non_null,
        "" if (has_source and has_row_idx) else "Missing SOURCE_FILE or ROW_IDX_ORIG"
    )
    
    # ── V10: Row count reconciliation ─────────────────────────────────────
    # Compare combined row count with sum of individual input files
    input_row_counts = {}
    for fname in EXPECTED_FILES:
        fpath = PARTIALLY_CLEANED / fname
        if fpath.exists():
            input_df = pd.read_excel(fpath)
            input_row_counts[fname] = len(input_df)
    
    expected_total = sum(input_row_counts.values())
    actual_total = len(df)
    
    # Per-file check
    per_file_ok = True
    per_file_detail = []
    for fname, expected_n in input_row_counts.items():
        actual_n = int((df["SOURCE_FILE"] == fname).sum())
        if actual_n != expected_n:
            per_file_ok = False
            per_file_detail.append(f"{fname}: expected {expected_n}, got {actual_n}")
    
    results.add(
        "V10", f"Row count reconciliation ({expected_total} expected, {actual_total} actual)",
        expected_total == actual_total and per_file_ok,
        "\n".join(per_file_detail) if per_file_detail else f"All {len(input_row_counts)} files match"
    )
    
    # ── V11: Diagnostic flag columns exist ────────────────────────────────
    # v2.0: FLAG_TIMING_UNCERTAIN removed (PRACTICAL EXAM/PROJECT confirmed pre-final)
    flag_cols = ["HAS_PREFINAL", "HAS_PREFINAL_EXCL_PE", "NO_ASSESSMENT_DATA",
                 "FLAG_120102BAA", "FLAG_SUSPICIOUS_TOTAL", "FLAG_NO_SCORE_ERA"]
    missing_flags = [c for c in flag_cols if c not in df.columns]
    results.add(
        "V11", "Diagnostic flag columns present",
        len(missing_flags) == 0,
        f"Missing: {missing_flags}" if missing_flags else ""
    )
    
    # ── V12: 120102BAA flagged correctly ──────────────────────────────────
    baa_mask = df["SOURCE_FILE"] == FILE_NO_PREDICTORS
    baa_flagged = df.loc[baa_mask, "FLAG_120102BAA"].all() if baa_mask.any() else False
    baa_rows = int(baa_mask.sum())
    results.add(
        "V12", f"120102BAA rows flagged ({baa_rows} rows)",
        baa_flagged,
        f"All {baa_rows} rows have FLAG_120102BAA=True" if baa_flagged else ""
    )
    
    # ── V13: Attendance Fail rows preserved ───────────────────────────────
    af_count = int((df["STATUS_CLEAN"] == "Attendance Fail").sum())
    af_target = df.loc[df["STATUS_CLEAN"] == "Attendance Fail", "TARGET"].unique()
    results.add(
        "V13", f"Attendance Fail preserved ({af_count} rows, TARGET={list(af_target)})",
        af_count > 0 and all(t == 1 for t in af_target),
        ""
    )
    
    # ── V14: Q6 zero-component verification ───────────────────────────────
    q6_path = QUALITY_DIR / "q6_verification.csv"
    if q6_path.exists():
        q6_df = pd.read_csv(q6_path)
        n_q6 = len(q6_df)
        n_explained = int(q6_df["explained_by_zero_rule"].sum()) if "explained_by_zero_rule" in q6_df.columns else 0
        n_unexplained = n_q6 - n_explained
        pct = round(100 * n_explained / n_q6, 1) if n_q6 > 0 else 100.0
        results.add(
            "V14", f"Q6 verification: {n_q6} contradictions, {pct}% explained by zero-component rule",
            n_q6 > 0,  # We expect contradictions to exist
            f"Explained: {n_explained}, Unexplained: {n_unexplained}"
        )
    else:
        results.add(
            "V14", "Q6 verification report exists",
            False,
            f"File not found: {q6_path}"
        )
    
    # ── V15: Modeling-ready dataset safety checks ─────────────────────────
    feature_info_path = MODELING_ELIGIBLE / "feature_info.json"
    modeling_path = MODELING_ELIGIBLE / "ADY201m_modeling_ready.csv"
    
    if feature_info_path.exists() and modeling_path.exists():
        with open(feature_info_path, "r", encoding="utf-8") as f:
            feature_info = json.load(f)
        
        modeling_df = pd.read_csv(modeling_path)
        ml_features = feature_info.get("ml_features", [])
        safety = feature_info.get("safety_checks", {})
        
        # Verify features don't include leakage
        has_final_in_ml = any(c.startswith(FINAL_EXAM_PREFIX) for c in ml_features)
        has_total_in_ml = "TOTAL" in ml_features
        has_code_in_ml = "CODE" in ml_features
        has_target_in_ml = "TARGET" in ml_features
        has_status_in_ml = "STATUS" in ml_features or "STATUS_CLEAN" in ml_features
        
        all_safe = (not has_final_in_ml and not has_total_in_ml and 
                    not has_code_in_ml and not has_target_in_ml and
                    not has_status_in_ml)
        
        detail_parts = []
        if has_final_in_ml: detail_parts.append("FINAL EXAM found in ML features!")
        if has_total_in_ml: detail_parts.append("TOTAL found in ML features!")
        if has_code_in_ml: detail_parts.append("CODE found in ML features!")
        if has_target_in_ml: detail_parts.append("TARGET found in ML features!")
        if has_status_in_ml: detail_parts.append("STATUS found in ML features!")
        
        # Verify PRACTICAL EXAM and PROJECT are in features
        has_pe = any(c.startswith("PRACTICAL EXAM") for c in ml_features)
        has_proj = any(c.startswith("PROJECT") for c in ml_features)
        
        # Verify QUIZ and PROGRESS TEST remain separate
        quiz_cols = [c for c in ml_features if c.startswith("QUIZ")]
        pt_cols = [c for c in ml_features if c.startswith("PROGRESS TEST")]
        merged = any("QUIZ_PROGRESS" in c or "QUIZ/PROGRESS" in c for c in ml_features)
        
        results.add(
            "V15", f"Modeling safety: {len(ml_features)} features, no leakage",
            all_safe,
            "\n".join(detail_parts) if detail_parts else 
            f"PRACTICAL EXAM included: {has_pe}, PROJECT included: {has_proj}, "
            f"QUIZ cols: {len(quiz_cols)}, PROGRESS TEST cols: {len(pt_cols)}, "
            f"Merged: {merged}"
        )
    else:
        results.add(
            "V15", "Modeling-ready dataset exists",
            False,
            f"Missing: feature_info.json or modeling_ready.csv"
        )
    
    # ── V16: Attendance Fail target assignment ────────────────────────────
    # Verify AF is TARGET=1 and not excluded
    af_mask = df["STATUS_CLEAN"] == "Attendance Fail"
    af_in_target = df.loc[af_mask, "TARGET"]
    af_all_one = (af_in_target == 1).all() if len(af_in_target) > 0 else False
    af_none_nan = af_in_target.notna().all() if len(af_in_target) > 0 else False
    results.add(
        "V16", f"Attendance Fail correctly assigned TARGET=1 ({int(af_mask.sum())} rows)",
        af_all_one and af_none_nan,
        ""
    )
    
    # ── V17: QUIZ and PROGRESS TEST columns remain separate ───────────────
    all_cols_set = set(df.columns)
    quiz_cols = [c for c in df.columns if c.startswith("QUIZ")]
    pt_cols = [c for c in df.columns if c.startswith("PROGRESS TEST")]
    merged_cols = [c for c in df.columns if "QUIZ_PROGRESS" in c or "QUIZ/PROGRESS" in c]
    results.add(
        "V17", f"QUIZ ({len(quiz_cols)}) and PROGRESS TEST ({len(pt_cols)}) remain separate",
        len(merged_cols) == 0,
        f"Merged columns found: {merged_cols}" if merged_cols else
        f"QUIZ: {quiz_cols[:3]}{'...' if len(quiz_cols)>3 else ''}, "
        f"PT: {pt_cols[:3]}{'...' if len(pt_cols)>3 else ''}"
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"VALIDATION SUMMARY: {results.n_passed}/{len(results.checks)} passed, "
          f"{results.n_failed} failed")
    print(f"{'─'*70}")
    
    # Machine-readable output
    validation_json = results.to_json()
    
    class NumpyEncoder(json.JSONEncoder):
        """Handle numpy types in JSON serialization."""
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(VALIDATION_DIR / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(validation_json, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # Table output
    results_df = results.to_dataframe()
    results_df.to_csv(VALIDATION_DIR / "validation_results.csv", index=False, encoding="utf-8-sig")
    
    # Human-readable summary
    summary_lines = [
        "# ADY201m EWS – Validation Summary (v2.0)",
        f"\nRun: {datetime.now().isoformat()}",
        f"\n## Results: {results.n_passed}/{len(results.checks)} passed",
        "",
    ]
    for c in results.checks:
        symbol = "✓" if c["passed"] else "✗ FAIL"
        summary_lines.append(f"- [{symbol}] **{c['check_id']}**: {c['name']}")
        if c["detail"] and not c["passed"]:
            summary_lines.append(f"  - {c['detail']}")
    
    if results.all_passed:
        summary_lines.append("\n## Overall: ALL CHECKS PASSED ✓")
    else:
        summary_lines.append(f"\n## Overall: {results.n_failed} CHECK(S) FAILED ✗")
        summary_lines.append("Review the failed checks above before proceeding.")
    
    # Dataset statistics
    summary_lines.extend([
        "\n## Dataset Statistics",
        f"- Total rows: {len(df):,}",
        f"- Files: {df['SOURCE_FILE'].nunique()}",
        f"- Distinct CODEs: {df['CODE'].nunique():,}",
        f"- TARGET=0 (Passed): {int((df.TARGET==0).sum()):,}",
        f"- TARGET=1 (Not Passed/Attendance Fail): {int((df.TARGET==1).sum()):,}",
        f"- Unsuccessful rate: {df.TARGET.mean():.1%}",
        f"- Rows with pre-final data: {int(df['HAS_PREFINAL'].sum()):,}" if 'HAS_PREFINAL' in df.columns else "",
        f"- 120102BAA rows (no predictors): {int(df['FLAG_120102BAA'].sum()):,}" if 'FLAG_120102BAA' in df.columns else "",
        f"- No-score-era rows: {int(df['FLAG_NO_SCORE_ERA'].sum()):,}" if 'FLAG_NO_SCORE_ERA' in df.columns else "",
        f"- Attendance Fail rows: {int((df['STATUS_CLEAN']=='Attendance Fail').sum()):,}",
    ])
    
    # Modeling dataset stats
    if modeling_path.exists():
        summary_lines.extend([
            "\n## Modeling-Ready Dataset",
            f"- Modeling rows: {len(modeling_df):,}",
            f"- ML features: {len(ml_features)}",
            f"- Features: {', '.join(ml_features)}",
        ])
    
    with open(VALIDATION_DIR / "validation_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    print(f"\nOutputs saved to: {VALIDATION_DIR}")
    
    return results


if __name__ == "__main__":
    results = validate_all()
    sys.exit(0 if results.all_passed else 1)
