"""Quick verification of key audit findings against actual Excel data."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import os

DATA = r"PARTIALLY CLEANED DATA"
files = sorted(os.listdir(DATA))
print(f"=== {len(files)} files found ===")

# 1. Verify 120102BAA structure
print("\n--- 120102BAA verification ---")
df = pd.read_excel(os.path.join(DATA, "120102BAA_TongHop.xlsx"))
print(f"Columns: {list(df.columns)}")
print(f"Rows: {len(df)}")
print(f"TOTAL not-null: {df.TOTAL.notna().sum()}")
print(f"TOTAL == 0 (among non-null): {(df.TOTAL.dropna() == 0).sum()}")
print(f"STATUS_CLEAN values: {df.STATUS_CLEAN.value_counts().to_dict()}")
print(f"TARGET values: {df.TARGET.value_counts().to_dict()}")

# 2. Verify STATUS_CLEAN/TARGET mapping across all files  
print("\n--- STATUS_CLEAN / TARGET cross-check ---")
all_dfs = []
for f in files:
    d = pd.read_excel(os.path.join(DATA, f))
    d["SOURCE_FILE"] = f
    all_dfs.append(d)
    
combined = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows: {len(combined)}")
print(f"Distinct STATUS_CLEAN: {sorted(combined.STATUS_CLEAN.unique())}")
print(f"STATUS_CLEAN x TARGET cross-tab:")
ct = pd.crosstab(combined.STATUS_CLEAN, combined.TARGET)
print(ct)

# 3. Check for truncated statuses
print("\n--- STATUS (raw) unique values ---")
print(sorted(combined.STATUS.unique()))

# 4. Verify score ranges
print("\n--- Score ranges ---")
id_meta = ["Mã học kỳ", "Mã lớp", "NO.", "CODE", "STATUS", "STATUS_CLEAN", "TARGET", "TOTAL"]
for f in files:
    d = pd.read_excel(os.path.join(DATA, f))
    score_cols = [c for c in d.columns if c not in id_meta]
    for c in score_cols:
        s = d[c].dropna()
        if len(s) > 0:
            if s.min() < 0 or s.max() > 10:
                print(f"  OUT OF RANGE: {f} {c}: min={s.min()}, max={s.max()}")

print("Score range check complete (no output = all within 0-10)")

# 5. Verify CODE repetition
print(f"\n--- CODE repetition ---")
print(f"Distinct CODEs: {combined.CODE.nunique()}")
code_counts = combined.CODE.value_counts()
print(f"Max rows per CODE: {code_counts.max()}")
print(f"CODEs appearing more than once: {(code_counts > 1).sum()}")

# 6. Verify columns present in all files
common_cols = set(all_dfs[0].columns) - {"SOURCE_FILE"}
for d in all_dfs[1:]:
    common_cols &= set(d.columns) - {"SOURCE_FILE"}
print(f"\nColumns in ALL 18 files: {sorted(common_cols)}")

# 7. Quick check per-file columns
print("\n--- Per-file assessment columns ---")
for f in files:
    d = pd.read_excel(os.path.join(DATA, f))
    score_cols = [c for c in d.columns if c not in id_meta]
    print(f"  {f[:12]:15s} ({len(d):6d} rows): {score_cols}")
