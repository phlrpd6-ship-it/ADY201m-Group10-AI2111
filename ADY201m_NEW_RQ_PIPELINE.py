# ============================================================
# ADY201m - Early Warning System (NEW RQ PIPELINE)
# ============================================================
# RQ1: Which pre-final assessment components are most consistently
#      associated with unsuccessful course outcomes across different
#      assessment structures?
# RQ2: Which tree-based ML model provides the best recall-oriented
#      early prediction using only pre-final assessment information?
# RQ3: To what extent does SMOTE + validation-based F2 threshold
#      optimization improve at-risk detection, and can SHAP provide
#      record-level explanations?
#
# Reproducibility notes
# - TARGET: 0 = successful, 1 = unsuccessful
# - FINAL EXAM and TOTAL are NEVER predictors
# - Imputer is fitted on TRAIN only
# - Model selection uses VALIDATION F2
# - SMOTE is applied to TRAIN only
# - Threshold optimization uses VALIDATION only
# - TEST is used only for final evaluation
# - Group split by CODE is enabled by default. Confirm that CODE is a
#   stable anonymized student identifier across source files.
# - SHAP explains model behavior, not causal effects.
# ============================================================

from pathlib import Path
import json
import warnings
import sys

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
import shap

RANDOM_STATE = 42
USE_GROUP_SPLIT = True
MIN_RQ1_N_WITHIN_FILE = 30
OUT_DIR = Path("ADY201m_NEW_RQ_RESULTS")
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
# Preferred: one already-combined CSV.
# Fallback: exactly 18 *_cleaned.csv files in the current folder.
COMBINED_CANDIDATES = [
    Path("ADY201m_18_CSV_COMBINED.csv"),
    Path("ADY201m_combined_cleaned.csv"),
]

combined_path = next((p for p in COMBINED_CANDIDATES if p.exists()), None)

if combined_path is not None:
    df = pd.read_csv(combined_path, low_memory=False)
    print(f"[OK] Loaded combined CSV: {combined_path}")
else:
    files = sorted([
        p for p in Path(".").glob("*_cleaned*.csv")
        if "COMBINED" not in p.name.upper()
        and "combined" not in p.name.lower()
    ])

    # Keep only one copy per logical source file if Windows added '(1)'.
    selected = {}
    for p in files:
        key = p.name.replace("_cleaned(1).csv", "_cleaned.csv")
        selected.setdefault(key, p)
    files = list(selected.values())

    if len(files) != 18:
        raise FileNotFoundError(
            "Expected ADY201m_18_CSV_COMBINED.csv or exactly 18 cleaned CSV files "
            f"in this folder, found {len(files)}."
        )

    frames = []
    for p in files:
        tmp = pd.read_csv(p, low_memory=False)
        if "SOURCE_FILE" not in tmp.columns:
            tmp["SOURCE_FILE"] = p.name
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True, sort=False)
    print("[OK] Loaded and unioned 18 cleaned CSV files")

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
# 3. RQ1 - CROSS-STRUCTURE CONSISTENCY
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
# 4. TRAIN / VALIDATION / TEST SPLIT
# ------------------------------------------------------------
X = model_df[FEATURES].copy()
y = model_df["TARGET"].copy()

if USE_GROUP_SPLIT and model_df["CODE"].notna().all():
    groups = model_df["CODE"].astype(str)
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
    split_strategy = "GroupShuffleSplit by CODE"
else:
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=RANDOM_STATE
    )
    split_strategy = "Stratified random split"

print("\n=== SPLIT ===")
for name, yy in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
    print(f"{name:10s}: {len(yy):,} | TARGET1={100*yy.mean():.2f}%")
print("Strategy:", split_strategy)

split_summary = pd.DataFrame([
    {"split": "train", "n": len(y_train), "target1_pct": 100*y_train.mean()},
    {"split": "validation", "n": len(y_val), "target1_pct": 100*y_val.mean()},
    {"split": "test", "n": len(y_test), "target1_pct": 100*y_test.mean()},
])
split_summary.to_csv(OUT_DIR / "01_split_summary.csv", index=False, encoding="utf-8-sig")

# Preserve missing mask for observed-only SHAP reporting later
X_test_original = X_test.copy()

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index)
X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=FEATURES, index=X_val.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES, index=X_test.index)

# ------------------------------------------------------------
# 5. RQ2 - FIVE MODEL COMPARISON
# ------------------------------------------------------------
def registry():
    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            random_state=RANDOM_STATE, verbosity=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
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

rq2_rows = []
for name, model in registry().items():
    print(f"Training {name}...")
    model.fit(X_train_imp, y_train)
    val_m = metrics(y_val, prob(model, X_val_imp), 0.50)
    test_m = metrics(y_test, prob(model, X_test_imp), 0.50)
    rq2_rows.append({
        "model": name,
        **{f"val_{k}": v for k, v in val_m.items()},
        **{f"test_{k}": v for k, v in test_m.items()},
    })

rq2 = pd.DataFrame(rq2_rows).sort_values(
    ["val_f2", "val_recall", "val_pr_auc"], ascending=False
).reset_index(drop=True)
best_name = rq2.iloc[0]["model"]
rq2.to_csv(OUT_DIR / "RQ2_model_comparison.csv", index=False, encoding="utf-8-sig")

print("\n=== RQ2: MODEL COMPARISON (selected by validation F2) ===")
print(rq2[[
    "model", "val_f2", "val_recall", "val_pr_auc",
    "test_precision", "test_recall", "test_f2", "test_roc_auc", "test_pr_auc"
]].to_string(index=False))
print("Selected model:", best_name)

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

smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)
smote_model = registry()[best_name]
smote_model.fit(X_train_sm, y_train_sm)
smote_val_p = prob(smote_model, X_val_imp)
smote_t, smote_thresholds = best_f2_threshold(y_val, smote_val_p)
smote_test_p = prob(smote_model, X_test_imp)
smote_m = metrics(y_test, smote_test_p, smote_t)

rq3 = pd.DataFrame([
    {"setting": f"{best_name} - baseline", **baseline_m},
    {"setting": f"{best_name} - SMOTE", **smote_m},
])
for k in ["precision", "recall", "f1", "f2", "roc_auc", "pr_auc"]:
    rq3[f"delta_{k}_vs_baseline"] = rq3[k] - rq3.loc[0, k]

rq3.to_csv(OUT_DIR / "RQ3_baseline_vs_smote.csv", index=False, encoding="utf-8-sig")
baseline_thresholds.to_csv(OUT_DIR / "RQ3_baseline_validation_thresholds.csv", index=False)
smote_thresholds.to_csv(OUT_DIR / "RQ3_smote_validation_thresholds.csv", index=False)

print("\n=== RQ3: BASELINE VS SMOTE ===")
print(rq3[["setting", "threshold", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc", "fp", "fn", "tp"]].round(4).to_string(index=False))

# ------------------------------------------------------------
# 7. SHAP - GLOBAL + OBSERVED-ONLY LOCAL EXPLANATIONS
# ------------------------------------------------------------
SHAP_OK = False
try:
    sample_n = min(500, len(X_test_imp))
    X_shap = X_test_imp.iloc[:sample_n].copy()
    X_shap_original = X_test_original.loc[X_shap.index].copy()

    explainer = shap.TreeExplainer(smote_model, feature_perturbation="interventional")
    raw = explainer.shap_values(X_shap, check_additivity=False)

    if isinstance(raw, list):
        values = np.asarray(raw[1])
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[:, :, 1]

    global_imp = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    global_imp.to_csv(OUT_DIR / "RQ3_SHAP_global_importance.csv", index=False, encoding="utf-8-sig")

    shap.summary_plot(values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "RQ3_SHAP_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Local examples: only report assessment features that were actually observed
    # in the original row before imputation. This avoids presenting structurally
    # missing/non-applicable assessments as real student-level drivers.
    local_rows = []
    for pos in range(min(10, len(X_shap))):
        idx = X_shap.index[pos]
        p = float(smote_test_p[X_test_imp.index.get_loc(idx)])
        pred = int(p >= smote_t)
        actual = int(y_test.loc[idx])
        code = str(model_df.loc[idx, "CODE"])

        for j, feat in enumerate(FEATURES):
            original_value = X_shap_original.iloc[pos, j]
            if pd.isna(original_value):
                continue
            sv = float(values[pos, j])
            local_rows.append({
                "test_row_index": int(idx),
                "code": code,
                "actual_target": actual,
                "predicted_target": pred,
                "predicted_probability": p,
                "feature": feat,
                "observed_feature_value": float(original_value),
                "shap_value": sv,
                "direction": "toward unsuccessful" if sv > 0 else "toward successful" if sv < 0 else "neutral",
                "abs_shap": abs(sv),
            })

    local = pd.DataFrame(local_rows)
    if not local.empty:
        local = local.sort_values(["test_row_index", "abs_shap"], ascending=[True, False])
        local.to_csv(OUT_DIR / "RQ3_SHAP_local_observed_only.csv", index=False, encoding="utf-8-sig")
        local.groupby("test_row_index", group_keys=False).head(10).to_csv(
            OUT_DIR / "RQ3_SHAP_local_observed_only_top10.csv", index=False, encoding="utf-8-sig"
        )
    SHAP_OK = True
    print("[OK] SHAP completed. Local output contains observed assessment features only.")
except Exception as e:
    print("[WARNING] SHAP failed:", repr(e))
    (OUT_DIR / "RQ3_SHAP_ERROR.txt").write_text(repr(e), encoding="utf-8")

# ------------------------------------------------------------
# 8. FINAL SUMMARY
# ------------------------------------------------------------
summary = []
summary.append("ADY201m NEW RQ PIPELINE SUMMARY")
summary.append("=" * 60)
summary.append(f"Rows all: {len(df):,}")
summary.append(f"Rows modeling: {len(model_df):,}")
summary.append(f"Split strategy: {split_strategy}")
summary.append("")

if not rq1_family.empty:
    broad = rq1_family[rq1_family["distinct_files"] >= 5]
    if not broad.empty:
        top_broad = broad.iloc[0]
        summary.append(
            "RQ1: Among assessment families replicated across at least 5 source files, "
            f"{top_broad['assessment_family']} had the largest median absolute within-file "
            f"correlation ({top_broad['median_abs_correlation']:.4f}) across "
            f"{int(top_broad['distinct_files'])} files. Association does not imply causation."
        )

sel = rq2.iloc[0]
summary.append(
    f"RQ2: {best_name} was selected by validation F2={sel['val_f2']:.4f}. "
    f"Test F2={sel['test_f2']:.4f}, recall={sel['test_recall']:.4f}, "
    f"precision={sel['test_precision']:.4f}."
)
summary.append(
    f"RQ3: Baseline F2={baseline_m['f2']:.4f}; SMOTE F2={smote_m['f2']:.4f}; "
    f"delta={smote_m['f2']-baseline_m['f2']:+.4f}. "
    f"Recall delta={smote_m['recall']-baseline_m['recall']:+.4f}."
)
summary.append(
    "SHAP: " + ("completed; local files show observed features only." if SHAP_OK else "not completed in this run.")
)
summary.append("")
summary.append("Caution: Confirm CODE semantics before making a student-level longitudinal claim.")
summary.append("Caution: Missing assessments are often structural and should not be treated as zero.")
summary.append("Caution: SHAP values explain model predictions, not causal root causes.")

summary_text = "\n".join(summary)
print("\n" + summary_text)
(OUT_DIR / "FINAL_NEW_RQ_SUMMARY.txt").write_text(summary_text, encoding="utf-8")

metadata = {
    "random_state": RANDOM_STATE,
    "use_group_split": USE_GROUP_SPLIT,
    "split_strategy": split_strategy,
    "features": FEATURES,
    "selected_model": best_name,
    "baseline_threshold": baseline_t,
    "smote_threshold": smote_t,
    "shap_success": SHAP_OK,
}
(OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(f"\n[DONE] Outputs saved in: {OUT_DIR.resolve()}")
