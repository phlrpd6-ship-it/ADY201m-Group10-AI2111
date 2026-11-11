"""Run the exact RQ1-RQ3 Pandas notebook analysis without Jupyter.
Run at the project root: py DEMO_PANDAS_RQ123.py
"""
import matplotlib
matplotlib.use('Agg')

# ------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chạy notebook từ thư mục gốc của ZIP sau giải nén.
ROOT = Path.cwd()
if not (ROOT/'DU_LIEU'/'18_CSV_GOC').exists():
    possible = list(ROOT.glob('*/DU_LIEU/18_CSV_GOC'))
    if len(possible) == 1:
        ROOT = possible[0].parent.parent
assert (ROOT/'DU_LIEU'/'18_CSV_GOC').exists(), 'Hãy mở notebook trong thư mục gốc chứa DU_LIEU.'
DATA = ROOT/'DU_LIEU'/'18_CSV_GOC'
RESULTS = ROOT/'KET_QUA_CHAY_V1_1'
OUT = ROOT/'KET_QUA_PANDAS_DEMO'
OUT.mkdir(exist_ok=True)
print('ROOT:',ROOT)

# ------------------------------------------------------------------
files = sorted(DATA.glob('*_cleaned.csv'))
assert len(files) == 18, f'Phải có đúng 18 CSV, hiện có {len(files)}'
parts = []
for file in files:
    part = pd.read_csv(file, low_memory=False)
    required = {'SOURCE_FILE', 'ROW_IDX_ORIG', 'CODE', 'TARGET', 'HAS_PREFINAL'}
    assert required.issubset(part.columns), f'{file.name} thiếu cột {required-set(part.columns)}'
    parts.append(part)
data = pd.concat(parts, ignore_index=True, sort=False)
assert len(data) == 124791
assert data['SOURCE_FILE'].nunique() == 18
assert not data.duplicated(['SOURCE_FILE','ROW_IDX_ORIG']).any()
assert data['TARGET'].isin([0,1]).all()
eligible = data.loc[data['HAS_PREFINAL'].astype(bool)].copy()
assert len(eligible) == 110704
print('Toàn bộ:',len(data), 'dòng | 18 nguồn | đủ điều kiện:',len(eligible))
print('Phân bổ nhãn trước SMOTE:')
print(eligible['TARGET'].value_counts().sort_index().rename('n').to_frame().assign(pct=lambda x:100*x.n/len(eligible)).round(2))

# ------------------------------------------------------------------
# Chỉ 21 biến điểm trước kỳ thi cuối kỳ dùng trong pipeline.
FEATURES = [
 'LABS-10','ASSIGNMENTS-10','PROGRESS TESTS-30','ASSIGNMENTS-30',
 'EXERCISES-30','QUIZ-20','ASSIGNMENTS-20','PRACTICAL EXAM-30',
 'PROGRESS TESTS-20','PROGRESS TESTS-10','QUIZ-10','LABS-15',
 'PRACTICAL EXAM-25','PRESENTATION-20','ACTIVE LEARNING-10','PROJECT-30',
 'EXERCISES-10','LABS-20','PRACTICAL EXAM-40','WORKSHOP-10','PRACTICAL EXAM-20'
]
assert len(FEATURES)==21 and set(FEATURES).issubset(eligible.columns)
per_file_rows=[]
for source, group in eligible.groupby('SOURCE_FILE'):
    for component in FEATURES:
        x = pd.to_numeric(group[component], errors='coerce')
        pair = pd.DataFrame({'score':x,'target':group['TARGET']}).dropna()
        if len(pair)<30 or pair.score.nunique()<2 or pair.target.nunique()<2:continue
        r = pair.score.corr(pair.target)
        if pd.notna(r):
            per_file_rows.append({'source_file':source, 'assessment_component':component,
                 'assessment_family':component.rsplit('-',1)[0], 'n':len(pair),
                 'pearson_r':r,'abs_r':abs(r)})
per_file = pd.DataFrame(per_file_rows)
rq1 = (per_file.groupby('assessment_family',as_index=False)
       .agg(distinct_files=('source_file','nunique'),component_file_pairs=('pearson_r','size'),
            median_abs_correlation=('abs_r','median'))
       .sort_values('median_abs_correlation',ascending=False))
known = pd.read_csv(RESULTS/'RQ1_cross_structure_consistency.csv')
check = rq1.merge(known[['assessment_family','median_abs_correlation']],on='assessment_family',suffixes=('_pandas','_pipeline'))
assert len(check)==len(rq1)==len(known)
assert np.allclose(check.median_abs_correlation_pandas,check.median_abs_correlation_pipeline,atol=1e-8)
print(rq1.round(6).to_string(index=False))
per_file.to_csv(OUT/'RQ1_pearson_per_source.csv',index=False)
rq1.to_csv(OUT/'RQ1_ranked_families.csv',index=False)
strong_broad=rq1.query('distinct_files >= 5').iloc[0]
print(f"RQ1 (>=5 files): {strong_broad.assessment_family}; median |r|={strong_broad.median_abs_correlation:.6f}; {int(strong_broad.distinct_files)} sources")
print('Lưu ý: PROJECT mạnh hơn về số tuyệt đối, nhưng chỉ có 2 file dữ liệu.')

# ------------------------------------------------------------------
chart=rq1.query('distinct_files >= 5').sort_values('median_abs_correlation')
ax=chart.plot.barh(x='assessment_family',y='median_abs_correlation',figsize=(9,4),legend=False)
ax.set(title='RQ1 | Median absolute Pearson r by assessment family (>=5 sources)',xlabel='Median |r|',ylabel='')
plt.tight_layout();plt.savefig(OUT/'RQ1_correlation.png',dpi=140);plt.close()

# ------------------------------------------------------------------
model = pd.read_csv(RESULTS/'RQ2_model_comparison.csv')
expected = {'Decision Tree','Random Forest','XGBoost','LightGBM','CatBoost'}
assert len(model)==5 and set(model.model)==expected
model['f2_recomputed'] = 5*model.val_tp / (5*model.val_tp+4*model.val_fn+model.val_fp)
assert np.allclose(model.val_f2, model.f2_recomputed,atol=1e-10)
model = model.sort_values('f2_recomputed',ascending=False).reset_index(drop=True)
print(model[['model','val_precision','val_recall','val_f2','val_pr_auc','test_f2']].round(6).to_string(index=False,na_rep='N/A'))
model.to_csv(OUT/'RQ2_5_models_checked.csv',index=False)
print(f'RQ2: {model.iloc[0].model}; validation F2 = {model.iloc[0].val_f2:.6f}')
print(f'     chênh với hạng 2 ({model.iloc[1].model}) = {model.iloc[0].val_f2-model.iloc[1].val_f2:.4f} (1 lần chia, 1 random_state; chỉ là bằng chứng mô tả)')

# ------------------------------------------------------------------
ax=model.sort_values('f2_recomputed').plot.barh(x='model',y='f2_recomputed',figsize=(9,4),legend=False)
ax.set(xlabel='Validation F2',ylabel='',title='RQ2 | Validation F2 across five algorithms',xlim=(0,1))
plt.tight_layout();plt.savefig(OUT/'RQ2_models.png',dpi=140);plt.close()

# ------------------------------------------------------------------
scenarios = pd.read_csv(RESULTS/'RQ3_four_scenarios.csv')
assert len(scenarios)==4
scenarios['precision_checked']=scenarios.tp/(scenarios.tp+scenarios.fp)
scenarios['recall_checked']=scenarios.tp/(scenarios.tp+scenarios.fn)
scenarios['f2_checked']=5*scenarios.tp/(5*scenarios.tp+4*scenarios.fn+scenarios.fp)
for metric in ('precision','recall','f2'):
    assert np.allclose(scenarios[metric],scenarios[metric+'_checked'],atol=1e-10)
print(scenarios[['scenario','threshold','precision_checked','recall_checked','f2_checked']].round(6).to_string(index=False))
scenarios.to_csv(OUT/'RQ3_4_scenarios_checked.csv',index=False)
s=scenarios.set_index('scenario')
b=s.loc['Baseline + F2 tuning']
m=s.loc['SMOTE + F2 tuning']
print(f"RQ3: delta F2 (SMOTE+tuning - Baseline+tuning) = {m.f2_checked-b.f2_checked:+.6f}")
print(f"     delta Recall = {m.recall_checked-b.recall_checked:+.6f}")
d_f2=m.f2_checked-b.f2_checked; d_rc=m.recall_checked-b.recall_checked
print(f"Kết luận (mô tả, 1 lần chia, chưa kiểm định thống kê): SMOTE+tuning {'cao hơn' if d_f2>0 else 'thấp hơn'} {abs(d_f2):.4f} F2 và {'cao hơn' if d_rc>0 else 'thấp hơn'} {abs(d_rc):.4f} recall so với baseline+tuning.")
if abs(d_f2)<0.005: print('Chênh lệch F2 rất nhỏ: không đủ cơ sở để kết luận SMOTE cải thiện (hay làm giảm) F2.')
u=s.loc['Baseline @ 0.50']
print(f"Chỉ tuning ngưỡng (baseline): F2 {b.f2_checked-u.f2_checked:+.6f}, recall {b.recall_checked-u.recall_checked:+.6f}, precision {b.precision_checked-u.precision_checked:+.6f}")

# ------------------------------------------------------------------
ax=scenarios.set_index('scenario')[['precision_checked','recall_checked','f2_checked']].plot.bar(
    figsize=(10,4.4),rot=15)
ax.set(ylim=(0,1),ylabel='Test metric',title='RQ3 | Four controlled scenarios (test set)')
plt.tight_layout();plt.savefig(OUT/'RQ3_scenarios.png',dpi=140);plt.close()
