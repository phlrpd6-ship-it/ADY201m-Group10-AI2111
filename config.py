"""
ADY201m EWS – Configuration
============================
Central configuration for the data-cleaning pipeline.
All paths, column definitions, and mapping rules live here.

Version: 2.0.0 — Incorporates methodological decisions (Q1–Q8).
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # ADY201m/

# ── Directory layout ─────────────────────────────────────────────────────
DATA_DIR            = PROJECT_ROOT / "data"
RAW_DIR             = DATA_DIR / "raw"
PARTIALLY_CLEANED   = DATA_DIR / "partially_cleaned"
CLEANED_DIR         = DATA_DIR / "cleaned"
MODELING_ELIGIBLE   = DATA_DIR / "modeling_eligible"
REPORTS_DIR         = PROJECT_ROOT / "reports"
MISSINGNESS_DIR     = REPORTS_DIR / "missingness"
SCHEMA_DIR          = REPORTS_DIR / "schema"
QUALITY_DIR         = REPORTS_DIR / "quality"
EXCLUSIONS_DIR      = REPORTS_DIR / "exclusions"
VALIDATION_DIR      = REPORTS_DIR / "validation"
SRC_DIR             = PROJECT_ROOT / "src"
NOTEBOOKS_DIR       = PROJECT_ROOT / "notebooks"

ALL_DIRS = [
    RAW_DIR, PARTIALLY_CLEANED, CLEANED_DIR, MODELING_ELIGIBLE,
    MISSINGNESS_DIR, SCHEMA_DIR, QUALITY_DIR, EXCLUSIONS_DIR,
    VALIDATION_DIR, NOTEBOOKS_DIR,
]

# ── Ensure directory structure exists ─────────────────────────────────────
def ensure_dirs():
    """Create all project directories if they don't exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)

# ── Identifier / metadata columns ────────────────────────────────────────
ID_COLS   = ["Mã học kỳ", "Mã lớp", "NO.", "CODE"]
META_COLS = ID_COLS + ["STATUS", "STATUS_CLEAN", "TARGET", "TOTAL"]

# Additional columns added during cleaning
ADDED_COLS = [
    "SOURCE_FILE",        # traceability to input file
    "ROW_IDX_ORIG",       # 0-based row index in the source file
    "HAS_PREFINAL",       # bool: at least one non-FINAL assessment score
    "HAS_PREFINAL_EXCL_PE",  # bool: at least one non-FINAL & non-PRACTICAL assessment
    "NO_ASSESSMENT_DATA", # bool: zero assessment columns populated
    "FLAG_120102BAA",     # bool: row from 120102BAA (no usable predictors)
    "FLAG_SUSPICIOUS_TOTAL",  # bool: known TOTAL issues (placeholders etc.)
    "FLAG_NO_SCORE_ERA",  # bool: row from a "no score data" schema era
    # FLAG_TIMING_UNCERTAIN removed: PRACTICAL EXAM and PROJECT confirmed
    # as pre-final assessments (methodological decision Q1).
]

# ── STATUS → TARGET mapping ──────────────────────────────────────────────
STATUS_TARGET_MAP = {
    "Passed":          0,
    "Not Passed":      1,
    "Attendance Fail": 1,
}

# Statuses that are excluded from modeling (TARGET = NaN)
STATUS_EXCLUDED = {"Suspended", "Exempt"}

# Known truncated status spellings and their canonical forms
# (verified against the audit: TOTAL/FINAL values are consistent)
TRUNCATED_STATUS_MAP = {
    "Passe":           "Passed",
    "Pass":            "Passed",
    "Not Passe":       "Not Passed",
    "Attendance Fai":  "Attendance Fail",
    "Attendance":      "Attendance Fail",
}

# ── Assessment column identification ─────────────────────────────────────
# Columns whose names start with these prefixes are FINAL EXAM columns (leakage)
FINAL_EXAM_PREFIX = "FINAL EXAM"

# Columns that are outcome / leakage-related — must NOT be used as predictors
LEAKAGE_COLS = ["TOTAL", "FINAL"]  # FINAL is a derived column in the combined dataset

# Confirmed pre-final assessment prefixes (Q1 resolved: these occur before FINAL)
PREFINAL_VALID_PREFIXES = ["PRACTICAL EXAM", "PROJECT"]

# Columns that must NEVER appear in the ML feature matrix
# (target-derived, post-outcome, identity, or leakage)
ML_EXCLUDE_COLS = [
    "CODE",               # anonymized student ID — not a predictor
    "TOTAL",              # outcome-related aggregate
    "TARGET",             # the modeling target itself
    "STATUS",             # raw status string
    "STATUS_CLEAN",       # cleaned status string — target-derived
    "SOURCE_FILE",        # course/file identity — metadata only
    "ROW_IDX_ORIG",       # traceability index
    "Mã học kỳ",          # semester — metadata
    "Mã lớp",             # class — metadata
    "NO.",                # row number — metadata
    "HAS_PREFINAL",       # diagnostic flag
    "HAS_PREFINAL_EXCL_PE",  # diagnostic flag
    "NO_ASSESSMENT_DATA", # diagnostic flag
    "FLAG_120102BAA",     # diagnostic flag
    "FLAG_SUSPICIOUS_TOTAL",  # diagnostic flag
    "FLAG_NO_SCORE_ERA",  # diagnostic flag
]

# ── Score validation ─────────────────────────────────────────────────────
SCORE_MIN = 0.0
SCORE_MAX = 10.0

# ── Files ────────────────────────────────────────────────────────────────
EXPECTED_FILES = [
    "010425BJA13_TongHop.xlsx",
    "030501BJA_TongHop.xlsx",
    "031904BJA_TongHop.xlsx",
    "031904BJC_TongHop.xlsx",
    "040116CIA13_TongHop.xlsx",
    "040209BJB_TongHop.xlsx",
    "091520AJA_TongHop.xlsx",
    "091520AJB_TongHop.xlsx",
    "120102BAA_TongHop.xlsx",
    "151907BJB_TongHop.xlsx",
    "160316BIA_TongHop.xlsx",
    "160616AIA_TongHop.xlsx",
    "161806AIB_TongHop.xlsx",
    "161806AIC_TongHop.xlsx",
    "161810CBA_TongHop.xlsx",
    "161810CJ24_TongHop.xlsx",
    "161810CJA_TongHop.xlsx",
    "161815AIB_TongHop.xlsx",
]

# ── File with no usable assessment predictors ────────────────────────────
FILE_NO_PREDICTORS = "120102BAA_TongHop.xlsx"
