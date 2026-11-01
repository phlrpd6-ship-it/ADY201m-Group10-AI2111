"""ADY201m: Demo Pandas cho RQ1, RQ2, RQ3.

Chay: py demo_pandas.py
- Doc truc tiep 18 CSV da clean (KHONG dung combined CSV).
- RQ1: tu tinh Pearson r theo tung file, sau do tinh median |r| theo nhom diem.
- RQ2: Pandas xep hang 5 model tu ket qua huan luyen DA LUU; tinh lai F2 tu TP/FP/FN.
- RQ3: Pandas kiem tra imbalance, tinh lai precision/recall/F2 va so sanh baseline/SMOTE.
- KHONG huan luyen lai model, khong chay SMOTE hay SHAP trong demo nay.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'DU_LIEU' / 'cleaned'
TRAINED = ROOT / 'DU_LIEU' / 'ADY201m_NEW_RQ_RESULTS_GROUPED'
METADATA = ROOT / 'DU_LIEU' / 'modeling_eligible' / 'feature_info.json'
OUTPUT = ROOT / 'KET_QUA'


def heading(title: str) -> None:
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def save_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUTPUT / filename, index=False, encoding='utf-8-sig', float_format='%.10g')
    print('  Saved:', 'KET_QUA/' + filename)


def f2_from_confusion(tp: pd.Series, fp: pd.Series, fn: pd.Series) -> pd.Series:
    # F2 = 5 TP / (5 TP + 4 FN + FP): recall quan trong gap 2 lan precision.
    return (5 * tp) / (5 * tp + 4 * fn + fp)


def load_18_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    heading('0. DOC 18 CSV BANG PANDAS (DU LIEU GOC CUA 3 RQ)')
    files = sorted(DATA.glob('*_cleaned.csv'))
    if len(files) != 18:
        raise FileNotFoundError(f'Can 18 file *_cleaned.csv, tim thay {len(files)} trong {DATA}')

    parts: list[pd.DataFrame] = []
    for file in files:
        part = pd.read_csv(file, low_memory=False)
        for col in ('SOURCE_FILE', 'ROW_IDX_ORIG', 'TARGET', 'HAS_PREFINAL', 'STATUS_CLEAN', 'CODE'):
            if col not in part:
                raise ValueError(f'{file.name}: thieu cot {col}')
        if part.empty or part['SOURCE_FILE'].nunique(dropna=False) != 1:
            raise ValueError(f'{file.name}: SOURCE_FILE khong dong nhat hoac file rong')
        print(f'  {file.name:30s}: {len(part):>6,} rows')
        parts.append(part)

    # Can chinh theo TEN COT; file khong co cot nao thi gia tri cua cot do la NaN.
    df = pd.concat(parts, ignore_index=True, sort=False)
    if df.duplicated(['SOURCE_FILE', 'ROW_IDX_ORIG']).any():
        raise ValueError('Trung cap (SOURCE_FILE, ROW_IDX_ORIG)')
    if len(df) != 124791 or df['SOURCE_FILE'].nunique() != 18:
        raise ValueError('Tong dong / so file khac voi metadata nguon')
    if not df['TARGET'].isin([0, 1]).all():
        raise ValueError('TARGET chi duoc nhan 0 hoac 1')
    if not df['HAS_PREFINAL'].isin([True, False]).all():
        raise ValueError('HAS_PREFINAL khong phai gia tri True/False')

    eligible = df.loc[df['HAS_PREFINAL']].copy()  # cung dieu kien voi v_ml_prefinal_dataset
    if len(eligible) != 110704:
        raise ValueError('So dong du dieu kien model khong dung 110704')
    overview = pd.DataFrame([{
        'total_rows': len(df),
        'source_files': df['SOURCE_FILE'].nunique(),
        'distinct_codes': df['CODE'].nunique(),
        'modeling_ready_rows': len(eligible),
    }])
    print('\n[Tuong ung SQL Query 1 - COUNT / DISTINCT]')
    print(overview.to_string(index=False))

    statuses = (df.groupby(['STATUS_CLEAN', 'TARGET'], dropna=False)
                .size().reset_index(name='n')
                .rename(columns={'STATUS_CLEAN': 'status_clean', 'TARGET': 'target'}))
    statuses['percent_of_all'] = (100 * statuses['n'] / len(df)).round(2)
    statuses = statuses.sort_values(['target', 'status_clean'])
    print('\n[Tuong ung SQL Query 2 - GROUP BY trang thai + TARGET]')
    print(statuses.to_string(index=False))
    save_csv(statuses, '01_status_by_target.csv')
    return df, eligible


def rq1_analysis(eligible: pd.DataFrame) -> None:
    heading('RQ1. DIEM THANH PHAN NAO LIEN HE MANH VOI KET QUA KHONG DAT?')
    # Bang SQL demo chi minh hoa TB PRACTICAL EXAM-30 theo TARGET.
    # Can loc HAS_PREFINAL=1 va score khong NULL cho giong view v_rq1_prefinal_long.
    sql_sample = eligible.dropna(subset=['PRACTICAL EXAM-30'])
    sql_equivalent = (sql_sample.groupby('TARGET')['PRACTICAL EXAM-30']
                      .agg(students_with_score='size', avg_practical_exam_30='mean')
                      .reset_index().rename(columns={'TARGET': 'target'}))
    sql_equivalent['avg_practical_exam_30'] = sql_equivalent['avg_practical_exam_30'].round(3)
    print('\n[Pandas tuong ung SQL Query 3 - AVG theo target]')
    print(sql_equivalent.to_string(index=False))
    save_csv(sql_equivalent, '02_RQ1_sql_avg_by_target.csv')

    info = json.loads(METADATA.read_text(encoding='utf-8'))
    features = [f for f in info['ml_features'] if f in eligible.columns]
    if len(features) != info['n_ml_features']:
        raise ValueError('Danh sach cot diem thieu so voi feature_info.json')

    rows = []
    # Tranh tinh pooled correlation giua cac cau truc diem khac nhau.
    for source, group in eligible.groupby('SOURCE_FILE'):
        for component in features:
            pair = group[[component, 'TARGET']].copy()
            pair[component] = pd.to_numeric(pair[component], errors='coerce')
            pair = pair.dropna()
            if len(pair) < 30 or pair[component].nunique() < 2 or pair['TARGET'].nunique() < 2:
                continue
            pearson_r = pair[component].corr(pair['TARGET'])
            if pd.notna(pearson_r):
                rows.append({
                    'source_file': source,
                    'assessment_component': component,
                    'assessment_family': component.rsplit('-', 1)[0],
                    'n': len(pair),
                    'correlation': float(pearson_r),
                    'absolute_correlation': abs(float(pearson_r)),
                })
    per_file = pd.DataFrame(rows)
    if per_file.empty:
        raise ValueError('RQ1 khong co cap diem-TARGET du dieu kien')
    summary = (per_file.groupby('assessment_family')
               .agg(distinct_files=('source_file', 'nunique'),
                    component_file_pairs=('source_file', 'size'),
                    median_abs_correlation=('absolute_correlation', 'median'))
               .reset_index()
               .sort_values('median_abs_correlation', ascending=False))

    # Kiem tra RQ1 co tinh dung nhu bo ket qua GROUPED da cung cap khong.
    known = pd.read_csv(TRAINED / 'RQ1_cross_structure_consistency.csv')
    aligned = summary[['assessment_family', 'median_abs_correlation']].merge(
        known[['assessment_family', 'median_abs_correlation']],
        on='assessment_family', how='outer', validate='one_to_one',
        suffixes=('_calculated', '_saved'), indicator=True)
    if (aligned['_merge'] != 'both').any() or not np.allclose(
            aligned['median_abs_correlation_calculated'], aligned['median_abs_correlation_saved'], atol=1e-8):
        raise ValueError('RQ1 tinh moi khong trung voi ket qua da luu')

    print('\n[Pandas: correlation theo tung SOURCE_FILE -> median |r| theo nhom]')
    print(summary.round(4).to_string(index=False))
    save_csv(per_file, '03_RQ1_correlation_per_file.csv')
    save_csv(summary, '04_RQ1_family_summary.csv')

    # Uu tien cac nhom xuat hien trong it nhat 5 file de bieu do on dinh hon.
    chart_data = summary.loc[summary['distinct_files'] >= 5].sort_values('median_abs_correlation')
    if chart_data.empty:
        raise ValueError('Khong co nhom RQ1 xuat hien >= 5 file')
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    bars = ax.barh(chart_data['assessment_family'], chart_data['median_abs_correlation'], color='#2873ae')
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xlim(0, 0.87)
    ax.set_xlabel('Median absolute Pearson correlation (within each file)')
    ax.set_title('RQ1 - Assessment groups appearing in at least 5 source files')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'RQ1_demo.png', dpi=160)
    plt.close(fig)
    print('Note: RQ1 la tuong quan, khong phai bang chung nhan qua.')


def rq2_analysis(eligible: pd.DataFrame) -> None:
    heading('RQ2. SO SANH 5 MO HINH DU DOAN (KET QUA DA TRAIN)')
    # Cung la 10 mau co PRACTICAL EXAM-30 nhu SQL demo query 4.
    cols = ['SOURCE_FILE', 'PRACTICAL EXAM-30', 'ASSIGNMENTS-30', 'PROGRESS TESTS-30', 'TARGET']
    sample = eligible.loc[eligible['PRACTICAL EXAM-30'].notna(), cols].head(10)
    print('\n[Pandas tuong ung SQL Query 4 - sample TOP 10 from modeling-ready view]')
    print(sample.to_string(index=False))
    raw = pd.read_csv(TRAINED / 'RQ2_model_comparison.csv')
    required = ['model', 'val_tp', 'val_fp', 'val_fn', 'val_f2', 'val_recall', 'val_pr_auc', 'test_f2']
    if len(raw) != 5 or not set(required).issubset(raw):
        raise ValueError('File RQ2 thieu model / metrics')
    ranked = raw.copy()
    # F2 kiem tra truc tiep tu ma tran nham lan luu san, khong goi day la huan luyen.
    ranked['val_f2_checked'] = f2_from_confusion(ranked['val_tp'], ranked['val_fp'], ranked['val_fn'])
    if not np.allclose(ranked['val_f2'], ranked['val_f2_checked'], atol=1e-10):
        raise ValueError('F2 cua model khong khop voi TP/FP/FN')
    ranked = ranked.sort_values('val_f2', ascending=False).reset_index(drop=True)
    output = ranked[['model', 'val_precision', 'val_recall', 'val_f2', 'val_f2_checked', 'val_pr_auc', 'test_f2']]
    print('\n[Pandas: tinh lai F2 tu confusion matrix va xep hang theo VALIDATION F2]')
    print(output.round(4).to_string(index=False, na_rep='N/A'))
    save_csv(output, '05_RQ2_model_comparison.csv')
    best = ranked.iloc[0]
    print(f"Selected by VALIDATION F2: {best['model']} ({best['val_f2']:.4f})")
    print(f"Its saved TEST F2: {best['test_f2']:.4f} (khong phai train moi)")

    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    plotted = ranked.sort_values('val_f2')
    bars = ax.barh(plotted['model'], plotted['val_f2'], color='#2873ae')
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9)
    ax.set_xlim(0, 1.04)
    ax.set_xlabel('Validation F2 - higher is better')
    ax.set_title('RQ2 - Comparing 5 trained models')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'RQ2_demo.png', dpi=160)
    plt.close(fig)


def rq3_analysis(eligible: pd.DataFrame) -> None:
    heading('RQ3. MAT CAN BANG LOP, SMOTE VS BASELINE, SHAP')
    distribution = eligible.groupby('TARGET').size().rename('n').reset_index().rename(columns={'TARGET': 'target'})
    distribution['pct'] = (100 * distribution['n'] / len(eligible)).round(2)
    print('\n[Pandas tuong ung SQL Query 5 - GROUP BY TARGET, % modeling ready]')
    print(distribution.to_string(index=False))
    save_csv(distribution, '06_RQ3_target_distribution.csv')

    raw = pd.read_csv(TRAINED / 'RQ3_baseline_vs_smote.csv')
    required = ['setting', 'tp', 'fp', 'fn', 'precision', 'recall', 'f2', 'threshold']
    if len(raw) != 2 or not set(required).issubset(raw):
        raise ValueError('File RQ3 khong dung schema')
    comparison = raw.copy()
    comparison['precision_checked'] = comparison['tp'] / (comparison['tp'] + comparison['fp'])
    comparison['recall_checked'] = comparison['tp'] / (comparison['tp'] + comparison['fn'])
    comparison['f2_checked'] = f2_from_confusion(comparison['tp'], comparison['fp'], comparison['fn'])
    for name in ('precision', 'recall', 'f2'):
        if not np.allclose(comparison[name], comparison[name + '_checked'], atol=1e-10):
            raise ValueError(f'{name} cua RQ3 khong khop ma tran nham lan')
    if 'baseline' not in comparison.iloc[0]['setting'].lower() or 'smote' not in comparison.iloc[1]['setting'].lower():
        raise ValueError('Thu tu baseline/SMOTE khong dung, khong the tinh delta')
    delta = comparison.iloc[1]['f2'] - comparison.iloc[0]['f2']
    output = comparison[['setting', 'threshold', 'precision', 'recall', 'f2', 'f2_checked']]
    print('\n[Pandas: tinh lai metrics tren TEST va so sanh hai setting da luu]')
    print(output.round(4).to_string(index=False))
    print(f'Delta TEST F2 (SMOTE - baseline): {delta:+.6f}')
    save_csv(output, '07_RQ3_baseline_vs_smote.csv')

    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    positions = np.arange(len(comparison))
    width = 0.24
    for j, (metric, color) in enumerate((('precision', '#2873ae'), ('recall', '#e7963d'), ('f2', '#2b9778'))):
        bars = ax.bar(positions + (j - 1) * width, comparison[metric], width=width, label=metric.title(), color=color)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(positions, ['Baseline', 'SMOTE'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Test-set metric')
    ax.set_title('RQ3 - Baseline and SMOTE (saved trained results)')
    ax.legend(loc='upper center', ncol=3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'RQ3_demo.png', dpi=160)
    plt.close(fig)

    shap = pd.read_csv(TRAINED / 'RQ3_SHAP_global_importance.csv')
    print('\n[Top 5 SHAP -- KET QUA DA TINH SAN; KHONG CHAY SHAP]')
    print(shap[['feature', 'mean_abs_shap']].head(5).round(4).to_string(index=False))
    print('SHAP goc:', 'DU_LIEU/ADY201m_NEW_RQ_RESULTS_GROUPED/RQ3_SHAP_summary.png')


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df, eligible = load_18_csv()
    del df
    rq1_analysis(eligible)
    rq2_analysis(eligible)
    rq3_analysis(eligible)
    heading('HOAN THANH: KET QUA O THU MUC KET_QUA/')
    print('RQ1 tinh bang Pandas tu du lieu; RQ2/RQ3 kiem tra va tong hop cac metrics da luu.')
    print('SQL Server khong can thiet de chay demo_pandas.py.')


if __name__ == '__main__':
    main()
