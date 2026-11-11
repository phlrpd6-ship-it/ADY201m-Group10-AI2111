# ============================================================
# ADY201m | PIPELINE CHI RQ1, RQ2, RQ3
# Revised from original ADY201m_NEW_RQ_PIPELINE_GROUPED.py
#   RQ1: Pre-final assessment components associated with failure
#   RQ2: Best of 5 models: DT, RF, XGBoost, LightGBM, CatBoost
#   RQ3: SMOTE + F2 threshold tuning vs baseline
# NO OTHER RESEARCH QUESTIONS ARE IMPLEMENTED.
# Data: original cleaned combined CSV; group by (SOURCE_FILE, CODE)
# Full mode retains original model parameters, split, and threshold grid.
# Quick-check mode is only for checking execution, NOT for reporting scores.
# ============================================================
from pathlib import Path
import json
import warnings
import sys
import platform
import argparse
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SCRIPT_VERSION = "ADY201m_PIPELINE_CHI_RQ1_RQ2_RQ3 v1.1 (no imputation; goc: GROUPED v1.0.0)"
SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Train va kiem tra RQ1, RQ2, RQ3")
parser.add_argument("--quick-check", action="store_true", help="Chay mau nho + mo hinh nho de thu code, KHONG dung so lieu bao cao")
args = parser.parse_args()
QUICK_CHECK = args.quick_check
if QUICK_CHECK:
    print("[DEMO NHANH] Du lieu lay mau, so cay it hon - KHONG dung ket qua de ket luan.")

RANDOM_STATE = 42
USE_GROUP_SPLIT = True
MIN_RQ1_N_WITHIN_FILE = 30
GROUP_KEY_DEFINITION = "SOURCE_FILE + '___' + CODE"
OUT_DIR = SCRIPT_DIR / ("KET_QUA_CHAY_NHANH" if QUICK_CHECK else "KET_QUA_CHAY_V1_1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. LOAD DATA - original combined cleaned CSV
# ------------------------------------------------------------
combined_path = SCRIPT_DIR / "DU_LIEU" / "ADY201m_combined_cleaned.csv"
if not combined_path.is_file():
    raise FileNotFoundError(f"Khong tim thay du lieu goc: {combined_path}")

df = pd.read_csv(combined_path, low_memory=False)
print(f"[OK] Loaded canonical combined CSV: {combined_path}")

# ------------------------------------------------------------
# 2. CORE DATA AUDIT
# ------------------------------------------------------------
required = {"TARGET", "CODE", "SOURCE_FILE", "HAS_PREFINAL"}
missing_required = required - set(df.columns)
if missing_required:
    raise ValueError(f"Missing required columns: {sorted(missing_required)}")

FEATURES = [
    "LABS-10",
    "ASSIGNMENTS-10",
    "PROGRESS TESTS-30",
    "ASSIGNMENTS-30",
    "EXERCISES-30",
    "QUIZ-20",
    "ASSIGNMENTS-20",
    "PRACTICAL EXAM-30",
    "PROGRESS TESTS-20",
    "PROGRESS TESTS-10",
    "QUIZ-10",
    "LABS-15",
    "PRACTICAL EXAM-25",
    "PRESENTATION-20",
    "ACTIVE LEARNING-10",
    "PROJECT-30",
    "EXERCISES-10",
    "LABS-20",
    "PRACTICAL EXAM-40",
    "WORKSHOP-10",
    "PRACTICAL EXAM-20",
]
FEATURES = [c for c in FEATURES if c in df.columns]

for bad in ["TOTAL", "FINAL EXAM-30", "FINAL EXAM-40"]:
    if bad in FEATURES:
        raise RuntimeError(f"Leakage column included in feature list: {bad}")

model_df = df[(df["HAS_PREFINAL"] == True) & df["TARGET"].isin([0, 1])].copy()
model_df["TARGET"] = model_df["TARGET"].astype(int)
if QUICK_CHECK:
    # Stratified sample only for execution testing. Final reported results
    # must be generated in full mode (without --quick-check).
    if len(model_df) > 6000:
        _, model_df = train_test_split(
            model_df, test_size=6000,
            stratify=model_df["TARGET"], random_state=RANDOM_STATE
        )
    model_df = model_df.sort_index().copy()


print("\n=== DATA AUDIT ===")
print(f"All rows:            {len(df):,}")
print(f"Union columns:       {len(df.columns):,}")
print(f"Distinct CODE:       {df['CODE'].nunique(dropna=True):,}")
print(f"Source files:        {df['SOURCE_FILE'].nunique(dropna=True):,}")
print(f"Modeling rows:       {len(model_df):,}")
print("Target all:")
print(df["TARGET"].value_counts(dropna=False).sort_index())
print("Target modeling:")
print(model_df["TARGET"].value_counts().sort_index())

# CODE recurrence across sources
code_source_n = (
    df.dropna(subset=["CODE", "SOURCE_FILE"])
      .groupby("CODE")["SOURCE_FILE"].nunique()
)
recurring_codes = int((code_source_n > 1).sum())
code_recurrence = pd.DataFrame({
    "measure": ["distinct_codes", "codes_in_multiple_source_files", "pct_recurring"],
    "value": [
        int(df["CODE"].nunique(dropna=True)),
        recurring_codes,
        100.0 * recurring_codes / max(1, df["CODE"].nunique(dropna=True)),
    ]
})
code_recurrence.to_csv(OUT_DIR / "00_CODE_recurrence.csv", index=False, encoding="utf-8-sig")

source_risk = (
    df[df["TARGET"].isin([0, 1])]
    .groupby("SOURCE_FILE")
    .agg(n=("TARGET", "size"), unsuccessful_rate=("TARGET", "mean"))
    .reset_index()
    .sort_values("unsuccessful_rate")
)
source_risk["unsuccessful_pct"] = 100 * source_risk["unsuccessful_rate"]
source_risk.to_csv(OUT_DIR / "00_source_file_risk.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 3. RQ1 - CROSS-STRUCTURE CONSISTENCY  (split-independent)
# ------------------------------------------------------------
def family_name(feature: str) -> str:
    # remove final numeric weight, e.g. PRACTICAL EXAM-30 -> PRACTICAL EXAM
    if "-" in feature and feature.rsplit("-", 1)[1].isdigit():
        return feature.rsplit("-", 1)[0]
    return feature

within_rows = []

for source_file, g in model_df.groupby("SOURCE_FILE"):
    for feat in FEATURES:
        if feat not in g.columns:
            continue
        sub = g[[feat, "TARGET"]].dropna()
        if len(sub) < MIN_RQ1_N_WITHIN_FILE:
            continue
        if sub[feat].nunique() < 2 or sub["TARGET"].nunique() < 2:
            continue
        corr = float(np.corrcoef(sub[feat].astype(float), sub["TARGET"].astype(float))[0, 1])
        within_rows.append({
            "source_file": source_file,
            "assessment_component": feat,
            "assessment_family": family_name(feat),
            "n": len(sub),
            "correlation": corr,
            "absolute_correlation": abs(corr),
            "avg_successful": float(sub.loc[sub.TARGET == 0, feat].mean()),
            "avg_unsuccessful": float(sub.loc[sub.TARGET == 1, feat].mean()),
        })

rq1_within = pd.DataFrame(within_rows)
rq1_within.to_csv(OUT_DIR / "RQ1_within_file_correlations.csv", index=False, encoding="utf-8-sig")

rq1_family = (
    rq1_within.groupby("assessment_family")
    .agg(
        distinct_files=("source_file", "nunique"),
        component_file_pairs=("source_file", "size"),
        total_n=("n", "sum"),
        median_correlation=("correlation", "median"),
        median_abs_correlation=("absolute_correlation", "median"),
        min_correlation=("correlation", "min"),
        max_correlation=("correlation", "max"),
    )
    .reset_index()
    .sort_values(["median_abs_correlation", "distinct_files"], ascending=[False, False])
)
rq1_family.to_csv(OUT_DIR / "RQ1_cross_structure_consistency.csv", index=False, encoding="utf-8-sig")
# Main RQ1 conclusion: require support from at least 5 distinct sources.
rq1_reliable = rq1_family[rq1_family["distinct_files"] >= 5].copy()
rq1_reliable.to_csv(OUT_DIR / "RQ1_top_assessment_families.csv", index=False, encoding="utf-8-sig")


print("\n=== RQ1: CROSS-STRUCTURE CONSISTENCY ===")
print(rq1_family.to_string(index=False))

plot_rq1 = rq1_family.sort_values("median_abs_correlation")
plt.figure(figsize=(9, 6))
plt.barh(plot_rq1["assessment_family"], plot_rq1["median_abs_correlation"])
plt.xlabel("Median |within-file correlation with unsuccessful outcome|")
plt.ylabel("Assessment family")
plt.title("RQ1 - Cross-structure consistency")
plt.tight_layout()
plt.savefig(OUT_DIR / "RQ1_cross_structure_consistency.png", dpi=180, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# 4. TRAIN / VALIDATION / TEST SPLIT  (composite group key)
# ------------------------------------------------------------
X = model_df[FEATURES].copy()
y = model_df["TARGET"].copy()

# Canonical grouping: (SOURCE_FILE, CODE). CODE is meaningful only
# within SOURCE_FILE (methodological decision Q3), so the composite
# key is the correct grouping unit. Global CODE grouping is NOT used.
group_key = (
    model_df["SOURCE_FILE"].astype(str) + "___" + model_df["CODE"].astype(str)
)

if (
    USE_GROUP_SPLIT
    and model_df["CODE"].notna().all()
    and model_df["SOURCE_FILE"].notna().all()
):
    groups = group_key
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    dev_pos, test_pos = next(gss1.split(X, y, groups=groups))

    X_dev = X.iloc[dev_pos].copy()
    y_dev = y.iloc[dev_pos].copy()
    groups_dev = groups.iloc[dev_pos].copy()
    X_test = X.iloc[test_pos].copy()
    y_test = y.iloc[test_pos].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_pos, val_pos = next(gss2.split(X_dev, y_dev, groups=groups_dev))
    X_train = X_dev.iloc[train_pos].copy()
    y_train = y_dev.iloc[train_pos].copy()
    X_val = X_dev.iloc[val_pos].copy()
    y_val = y_dev.iloc[val_pos].copy()
    split_strategy = "GroupShuffleSplit by (SOURCE_FILE, CODE)"
else:
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=RANDOM_STATE
    )
    split_strategy = "Stratified random split"

# Group-level audit on the composite key (works for both branches).
train_index = X_train.index
val_index = X_val.index
test_index = X_test.index
train_groups = set(group_key.loc[train_index])
val_groups = set(group_key.loc[val_index])
test_groups = set(group_key.loc[test_index])
group_overlap = {
    "train_validation": len(train_groups & val_groups),
    "train_test": len(train_groups & test_groups),
    "validation_test": len(val_groups & test_groups),
}

print("\n=== SPLIT ===")
for name, yy in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
    print(f"{name:10s}: {len(yy):,} | TARGET1={100*yy.mean():.2f}%")
print("Strategy:", split_strategy)
print(
    "Groups (train/val/test): "
    f"{len(train_groups):,} / {len(val_groups):,} / {len(test_groups):,}"
)
print("Group overlap (train-val/train-test/val-test):", group_overlap)

split_summary = pd.DataFrame([
    {"split": "train", "n": len(y_train), "target1_pct": 100*y_train.mean()},
    {"split": "validation", "n": len(y_val), "target1_pct": 100*y_val.mean()},
    {"split": "test", "n": len(y_test), "target1_pct": 100*y_test.mean()},
])
split_summary.to_csv(OUT_DIR / "01_split_summary.csv", index=False, encoding="utf-8-sig")

# Deterministic split artifact: exact row indices + composite group key,
# so the split can be independently reconstructed.
split_indices = pd.concat([
    pd.DataFrame({
        "index": train_index,
        "split": "train",
        "group_key": group_key.loc[train_index].values,
    }),
    pd.DataFrame({
        "index": val_index,
        "split": "validation",
        "group_key": group_key.loc[val_index].values,
    }),
    pd.DataFrame({
        "index": test_index,
        "split": "test",
        "group_key": group_key.loc[test_index].values,
    }),
], ignore_index=True)
split_indices.to_csv(OUT_DIR / "split_indices.csv", index=False, encoding="utf-8-sig")

# NO IMPUTATION (v1.1). Missing assessment scores are kept as NaN because
# each course only records some assessment components. DT/RF (scikit-learn
# >= 1.4), XGBoost, LightGBM and CatBoost handle NaN natively.
# SMOTE cannot process NaN, so ONLY the SMOTE scenarios use a median
# imputer fitted on the training partition (see section 6). All
# non-SMOTE results (RQ2 and the baseline in RQ3) use the raw NaN data.
X_train_imp, X_val_imp, X_test_imp = X_train, X_val, X_test

# ------------------------------------------------------------
# 5. RQ2 - FIVE MODEL COMPARISON
# ------------------------------------------------------------
# Convention: all five models are compared on VALIDATION only.
# The best model is selected by validation F2, and the untouched
# TEST set is used ONCE, for the SELECTED model only. Test metrics
# are never used to select, tune, or optimize anything.
def registry():
    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=30 if QUICK_CHECK else 300, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=30 if QUICK_CHECK else 300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=30 if QUICK_CHECK else 300, learning_rate=0.05,
            random_state=RANDOM_STATE, verbosity=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=30 if QUICK_CHECK else 300, depth=6, learning_rate=0.05,
            random_seed=RANDOM_STATE, verbose=False
        ),
    }


def prob(model, XX):
    return model.predict_proba(XX)[:, 1]


def metrics(y_true, p, threshold=0.50):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

fitted = {}
rq2_rows = []
for name, model in registry().items():
    print(f"Training {name}...")
    model.fit(X_train_imp, y_train)
    fitted[name] = model
    val_m = metrics(y_val, prob(model, X_val_imp), 0.50)
    rq2_rows.append({
        "model": name,
        **{f"val_{k}": v for k, v in val_m.items()},
    })

rq2 = pd.DataFrame(rq2_rows).sort_values(
    ["val_f2", "val_recall", "val_pr_auc"], ascending=False
).reset_index(drop=True)
best_name = rq2.iloc[0]["model"]

# Evaluate the SELECTED model ONCE on the untouched test set.
# Test metrics are reported for the selected model only; the other
# models' test metrics are intentionally not computed (no test-based
# selection, tuning, or optimization).
best_test_m = metrics(y_test, prob(fitted[best_name], X_test_imp), 0.50)
for k, v in best_test_m.items():
    rq2.loc[rq2["model"] == best_name, f"test_{k}"] = v

rq2.to_csv(OUT_DIR / "RQ2_model_comparison.csv", index=False, encoding="utf-8-sig")

print("\n=== RQ2: MODEL COMPARISON (selected by validation F2) ===")
print(rq2[["model", "val_f2", "val_recall", "val_pr_auc"]].to_string(index=False))
print(
    f"Selected model: {best_name} | "
    f"test F2={best_test_m['f2']:.4f}, recall={best_test_m['recall']:.4f}, "
    f"precision={best_test_m['precision']:.4f} (test evaluated once, selected model only)"
)

# ------------------------------------------------------------
# 6. RQ3 - BASELINE VS SMOTE + VALIDATION F2 THRESHOLD
# ------------------------------------------------------------
def best_f2_threshold(y_true, p):
    rows = []
    for t in np.arange(0.05, 0.951, 0.01):
        pred = (p >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        })
    tbl = pd.DataFrame(rows)
    best = tbl.sort_values(["f2", "recall", "precision"], ascending=False).iloc[0]
    return float(best["threshold"]), tbl

baseline = registry()[best_name]
baseline.fit(X_train_imp, y_train)
baseline_val_p = prob(baseline, X_val_imp)
baseline_t, baseline_thresholds = best_f2_threshold(y_val, baseline_val_p)
baseline_test_p = prob(baseline, X_test_imp)
baseline_m = metrics(y_test, baseline_test_p, baseline_t)

# SMOTE branch only: SMOTE requires complete data, so a median imputer
# fitted on TRAIN is applied to train/val/test for these scenarios.
smote_imputer = SimpleImputer(strategy="median")
X_train_si = pd.DataFrame(smote_imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index)
X_val_si = pd.DataFrame(smote_imputer.transform(X_val), columns=FEATURES, index=X_val.index)
X_test_si = pd.DataFrame(smote_imputer.transform(X_test), columns=FEATURES, index=X_test.index)
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_si, y_train)
smote_model = registry()[best_name]
smote_model.fit(X_train_sm, y_train_sm)
smote_val_p = prob(smote_model, X_val_si)
smote_t, smote_thresholds = best_f2_threshold(y_val, smote_val_p)
smote_test_p = prob(smote_model, X_test_si)
smote_m = metrics(y_test, smote_test_p, smote_t)

# Supplementary control (NOT one of the four scenarios): same median-imputed
# data as the SMOTE branch but WITHOUT SMOTE, to separate the effect of
# imputation from the effect of SMOTE.
ctrl_model = registry()[best_name]
ctrl_model.fit(X_train_si, y_train)
ctrl_t, _ = best_f2_threshold(y_val, prob(ctrl_model, X_val_si))
ctrl_test_p = prob(ctrl_model, X_test_si)
pd.DataFrame([
    {"scenario": "Imputed baseline @ 0.50", **metrics(y_test, ctrl_test_p, 0.50)},
    {"scenario": "Imputed baseline + F2 tuning", **metrics(y_test, ctrl_test_p, ctrl_t)},
]).to_csv(OUT_DIR / "RQ3_control_imputed_no_smote.csv", index=False, encoding="utf-8-sig")

rq3 = pd.DataFrame([
    {"setting": f"{best_name} - baseline", **baseline_m},
    {"setting": f"{best_name} - SMOTE", **smote_m},
])
for k in ["precision", "recall", "f1", "f2", "roc_auc", "pr_auc"]:
    rq3[f"delta_{k}_vs_baseline"] = rq3[k] - rq3.loc[0, k]

rq3.to_csv(OUT_DIR / "RQ3_baseline_vs_smote.csv", index=False, encoding="utf-8-sig")
baseline_thresholds.to_csv(OUT_DIR / "RQ3_baseline_validation_thresholds.csv", index=False)
smote_thresholds.to_csv(OUT_DIR / "RQ3_smote_validation_thresholds.csv", index=False)

# Additional RQ3 ablation: isolate threshold tuning vs SMOTE.
# This evaluation never uses test for model/threshold selection.
ablations = pd.DataFrame([
    {"scenario": "Baseline @ 0.50", **metrics(y_test, baseline_test_p, 0.50)},
    {"scenario": "Baseline + F2 tuning", **baseline_m},
    {"scenario": "SMOTE @ 0.50", **metrics(y_test, smote_test_p, 0.50)},
    {"scenario": "SMOTE + F2 tuning", **smote_m},
])
ablations.to_csv(OUT_DIR / "RQ3_four_scenarios.csv", index=False, encoding="utf-8-sig")

print("\n=== RQ3: BASELINE VS SMOTE ===")
print(rq3[["setting", "threshold", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc", "fp", "fn", "tp"]].round(4).to_string(index=False))


# ------------------------------------------------------------
# 7. CHARTS AND SUMMARY - ONLY 3 RQs
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.barh(rq2["model"][::-1], rq2["val_f2"][::-1])
plt.xlim(0, 1)
plt.xlabel("Validation F2 at 0.50")
plt.title("RQ2: Five-model comparison")
plt.tight_layout()
plt.savefig(OUT_DIR / "RQ2_model_comparison.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
plt.bar(ablations["scenario"], ablations["f2"])
plt.ylim(0, 1)
plt.ylabel("Test F2")
plt.title("RQ3: SMOTE and F2 threshold tuning")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "RQ3_four_scenarios.png", dpi=150)
plt.close()

summary = [
    "ADY201m - KET QUA PIPELINE CHI RQ1, RQ2, RQ3",
    "=" * 58,
    f"Version: {SCRIPT_VERSION}",
    f"Mode: {'QUICK-CHECK - DO NOT REPORT' if QUICK_CHECK else 'FULL ORIGINAL PARAMETERS'}",
    f"Data: {combined_path.name} | total_rows={len(df)} | modeling_rows={len(model_df)}",
    f"Split: {split_strategy} | overlap={group_overlap}",
]
if not rq1_reliable.empty:
    winner = rq1_reliable.iloc[0]
    summary.append(
        f"RQ1: {winner['assessment_family']} highest median |Pearson r|="
        f"{winner['median_abs_correlation']:.4f} in {int(winner['distinct_files'])} sources (>=5 sources)."
    )
else:
    summary.append("RQ1: No feature family replicated in at least 5 sources; no stable winner.")
summary += [
    f"RQ2: Best model={best_name}; validation F2={float(rq2.iloc[0]['val_f2']):.4f}; selected model test F2@0.50={best_test_m['f2']:.4f}.",
    f"RQ3: Baseline+tuning F2={baseline_m['f2']:.4f}, SMOTE+tuning F2={smote_m['f2']:.4f}, delta={smote_m['f2'] - baseline_m['f2']:+.4f}.",
    f"RQ3: Recall baseline={baseline_m['recall']:.4f}, SMOTE={smote_m['recall']:.4f}, delta={smote_m['recall']-baseline_m['recall']:+.4f}.",
    "Notes: Associations do not imply causation. Thresholds optimized using validation only.",
    "Notes: No imputation except inside the SMOTE scenarios (SMOTE cannot handle NaN).",
    "Notes: Class 1=unsuccessful. Scores from quick-check mode are NOT for the final report.",
]
(OUT_DIR / "TONG_KET_RQ1_RQ2_RQ3.txt").write_text("\n".join(summary), encoding="utf-8")
print("\n" + "\n".join(summary))

# Sufficient metadata to trace real training and reproducibility.
def version_of(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not found"

def serializable(x):
    if isinstance(x, dict):
        return {str(k): serializable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)):
        return [serializable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x

metadata={
    "script": SCRIPT_VERSION,
    "mode": "quick-check" if QUICK_CHECK else "full",
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "python_version": platform.python_version(),
    "package_versions": {p: version_of(p) for p in ["numpy","pandas","scikit-learn","imbalanced-learn","xgboost","lightgbm","catboost","matplotlib"]},
    "input_path": str(combined_path),
    "rows_total": len(df), "rows_model": len(model_df),
    "target_definition": {"0": "successful", "1": "unsuccessful"},
    "predictors": FEATURES,
    "excluded_leakage_features": ["FINAL EXAM", "FINAL EXAM-30", "FINAL EXAM-40", "TOTAL", "STATUS", "STATUS_CLEAN", "TARGET"],
    "split": split_strategy,
    "split_group_overlap": group_overlap,
    "split_sizes": {"train":len(y_train), "validation":len(y_val), "test":len(y_test)},
    "model_parameters": {k:repr(v) for k,v in registry().items()},
    "selected_model": best_name,
    "validation_f2_at_0_5": float(rq2.iloc[0]["val_f2"]),
    "threshold_search": "0.05 to 0.95 inclusive, step 0.01 on validation",
    "baseline_threshold": baseline_t,
    "smote_threshold": smote_t,
    "baseline_test_metrics": baseline_m,
    "smote_test_metrics": smote_m,
    "random_state": RANDOM_STATE,
    "imputation": "none for RQ2 and baseline (native NaN handling); median imputer fitted on train used ONLY in the SMOTE scenarios",
}
(OUT_DIR / "run_metadata.json").write_text(json.dumps(serializable(metadata),ensure_ascii=False,indent=2), encoding="utf-8")
print(f"\n[OK] Da xuat ket qua: {OUT_DIR}")

