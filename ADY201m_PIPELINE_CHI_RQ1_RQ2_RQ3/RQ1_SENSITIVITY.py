"""RQ1 sensitivity analysis: two-stage aggregation (additive; does not modify existing outputs).
Stage 0: same filters as DEMO_PANDAS_RQ123.py (n>=30, >=2 distinct scores, both classes).
Stage 1: median |r| over components within (SOURCE_FILE, family).
Stage 2: median across SOURCE_FILE.
Run: python RQ1_SENSITIVITY.py"""
from pathlib import Path
import pandas as pd, numpy as np
BASE = Path(__file__).resolve().parent
CSV = BASE/'DU_LIEU'/'18_CSV_GOC'
OUT = BASE/'KET_QUA_SENSITIVITY_RQ1'; OUT.mkdir(exist_ok=True)
FEATURES = ['LABS-10','ASSIGNMENTS-10','PROGRESS TESTS-30','ASSIGNMENTS-30','EXERCISES-30','QUIZ-20',
 'ASSIGNMENTS-20','PRACTICAL EXAM-30','PROGRESS TESTS-20','PROGRESS TESTS-10','QUIZ-10','LABS-15',
 'PRACTICAL EXAM-25','PRESENTATION-20','ACTIVE LEARNING-10','PROJECT-30','EXERCISES-10','LABS-20',
 'PRACTICAL EXAM-40','WORKSHOP-10','PRACTICAL EXAM-20']
data = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(CSV.glob('*_cleaned.csv'))], ignore_index=True, sort=False)
elig = data.loc[data['HAS_PREFINAL'].astype(bool)]
assert len(data)==124791 and len(elig)==110704
rows=[]
for src,g in elig.groupby('SOURCE_FILE'):
    for c in FEATURES:
        p=pd.DataFrame({'s':pd.to_numeric(g[c],errors='coerce'),'t':g['TARGET']}).dropna()
        if len(p)<30 or p.s.nunique()<2 or p.t.nunique()<2: continue
        r=p.s.corr(p.t)
        if pd.notna(r): rows.append((src,c,c.rsplit('-',1)[0],len(p),r,abs(r)))
pf=pd.DataFrame(rows,columns=['source_file','component','family','n','r','abs_r'])
orig=(pf.groupby('family').agg(files=('source_file','nunique'),pairs=('abs_r','size'),median_abs_r=('abs_r','median'))
        .sort_values('median_abs_r',ascending=False))
orig['rank']=range(1,len(orig)+1)
fam_file=pf.groupby(['family','source_file']).agg(components=('abs_r','size'),median_abs_r=('abs_r','median')).reset_index()
sens=(fam_file.groupby('family').agg(files=('source_file','nunique'),median_of_file_medians=('median_abs_r','median'))
        .sort_values('median_of_file_medians',ascending=False))
sens['rank']=range(1,len(sens)+1)
cmp=orig.join(sens,lsuffix='_original',rsuffix='_sensitivity')
cmp['rank_shift']=cmp['rank_original']-cmp['rank_sensitivity']
cmp['abs_diff']=cmp.median_of_file_medians-cmp.median_abs_r
cmp['multi_variant_files']=fam_file[fam_file.components>1].groupby('family').size().reindex(cmp.index).fillna(0).astype(int)
cmp=cmp.sort_values('rank_original')
fam_file.to_csv(OUT/'RQ1_family_by_file_median.csv',index=False)
cmp.round(6).to_csv(OUT/'RQ1_original_vs_sensitivity.csv')
print(cmp.round(4).to_string())
b=lambda d,col:d.query('files>=5').sort_values(col,ascending=False).index[0]
print('Leading family (>=5 files): original =',b(orig,'median_abs_r'),'| sensitivity =',b(sens,'median_of_file_medians'))
from scipy.stats import spearmanr
print('Spearman rank corr (all families):',round(spearmanr(cmp.median_abs_r,cmp.median_of_file_medians)[0],3))
