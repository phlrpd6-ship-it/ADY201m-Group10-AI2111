# Kiem thu full pipeline RQ1-RQ2-RQ3

- Full run completed successfully using 110,704 modeling rows out of 124,791 total rows (18 sources).
- Python 3.13.5; NumPy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0; imbalanced-learn 0.14.1; XGBoost 3.1.3; LightGBM 4.6.0; CatBoost 1.2.8.
- GroupShuffleSplit by (SOURCE_FILE,CODE), with ZERO group overlap between train/validation/test; split assignments matched the archived original's split_indices.csv exactly.

| Question | Archived original run (different library versions) | New verified full run |
|---|---|---|
| RQ1: highest supported family | PRACTICAL EXAM, median abs(r)=0.7193 | PRACTICAL EXAM, median abs(r)=0.7193 |
| RQ2: selected model | Random Forest, validation F2=0.8540 | LightGBM, validation F2=0.8462 |
| RQ3: tuned baseline F2 | 0.8408 | 0.8574 |
| RQ3: tuned SMOTE F2 | 0.8433 | 0.8553 |
| RQ3: tuned baseline recall | 0.8677 | 0.8846 |
| RQ3: tuned SMOTE recall | 0.8563 | 0.8758 |

IMPORTANT: This package does NOT claim the archived Random Forest selection was reproduced. Training outcomes differ although data source and group splits matched; installed library versions differ substantially and are a plausible cause. For your report, present results from the environment you actually run, and do not mix archived numbers with new output.

Four-scenario test F2 (new full run): baseline@0.50 0.8269; baseline+tuning 0.8574; SMOTE@0.50 0.8440; SMOTE+tuning 0.8553. SMOTE+tuning does NOT outperform baseline+tuning on the new full run.

All new outputs are in `KET_QUA_KIEM_THU_FULL/`; `KET_QUA_GOC_DE_THAM_KHAO/` contains old CSVs for reference only. Re-run on your own machine to generate `KET_QUA_CHAY_MOI/`.
