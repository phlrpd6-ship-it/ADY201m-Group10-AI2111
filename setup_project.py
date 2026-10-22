"""
ADY201m EWS – Project Setup
============================
Sets up the project directory structure and copies files to their proper locations.
Does NOT modify source files.

Usage:
    py src/setup_project.py
"""
import sys, io, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROJECT_ROOT, RAW_DIR, PARTIALLY_CLEANED, QUALITY_DIR,
    ensure_dirs,
)

WORKSPACE = PROJECT_ROOT.parent  # c:\PHAMHOANGLONG\HOC\ADY


def main():
    print("ADY201m EWS – Project Setup")
    print("="*50)
    
    ensure_dirs()
    
    # ── 1. Copy partially cleaned xlsx to data/partially_cleaned/ ─────────
    src_partial = WORKSPACE / "PARTIALLY CLEANED DATA"
    if src_partial.exists():
        n = 0
        for f in sorted(src_partial.glob("*.xlsx")):
            dest = PARTIALLY_CLEANED / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                n += 1
            else:
                # Verify same size
                if f.stat().st_size != dest.stat().st_size:
                    print(f"  WARNING: {f.name} differs from existing copy!")
        print(f"Partially cleaned data: copied {n} new files to {PARTIALLY_CLEANED}")
    else:
        print(f"WARNING: Source not found: {src_partial}")
    
    # ── 2. Copy raw data to data/raw/ ─────────────────────────────────────
    src_raw = WORKSPACE / "RAW DATA"
    if src_raw.exists():
        n = 0
        for f in sorted(src_raw.glob("*.xlsx")):
            dest = RAW_DIR / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                n += 1
        print(f"Raw data: copied {n} new files to {RAW_DIR}")
    else:
        print(f"WARNING: Source not found: {src_raw}")
    
    # ── 3. Copy audit artifacts to reports/quality/ ───────────────────────
    audit_report = WORKSPACE / "ADY201m_audit_report.md"
    audit_script = WORKSPACE / "audit_ady201m.py"
    audit_tables = WORKSPACE / "audit_tables"
    
    audit_dest = QUALITY_DIR / "audit"
    audit_dest.mkdir(parents=True, exist_ok=True)
    
    for src in [audit_report, audit_script]:
        if src.exists():
            dest = audit_dest / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
                print(f"  Copied {src.name} → {audit_dest}")
    
    if audit_tables.exists():
        dest_tables = audit_dest / "audit_tables"
        if not dest_tables.exists():
            shutil.copytree(audit_tables, dest_tables)
            print(f"  Copied audit_tables/ → {dest_tables}")
    
    print(f"\nProject structure ready at: {PROJECT_ROOT}")
    
    # ── 4. Show structure ─────────────────────────────────────────────────
    print("\nDirectory structure:")
    for p in sorted(PROJECT_ROOT.rglob("*")):
        if p.is_dir():
            rel = p.relative_to(PROJECT_ROOT)
            depth = len(rel.parts)
            print(f"  {'  '*depth}📁 {rel.name}/")


if __name__ == "__main__":
    main()
