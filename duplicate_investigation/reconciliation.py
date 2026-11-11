from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parent.parent
raw=root/'raw'; cleaned=root/'cleaned'; out=Path(__file__).resolve().parent
rows=[]
for p in sorted(cleaned.glob('*_cleaned.csv')):
 if p.name=='ADY201m_combined_cleaned.csv': continue
 c=pd.read_csv(p,low_memory=False)
 src=c.SOURCE_FILE.dropna().iloc[0]
 r=pd.read_excel(raw/src)
 rows.append({'SOURCE_FILE':src,'raw_rows':len(r),'cleaned_rows':len(c),'difference':len(r)-len(c),'row_idx_unique':not c.duplicated(['SOURCE_FILE','ROW_IDX_ORIG']).any(),'row_idx_min':c.ROW_IDX_ORIG.min(),'row_idx_max':c.ROW_IDX_ORIG.max()})
pd.DataFrame(rows).to_csv(out/'raw_cleaned_reconciliation.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(rows).to_string(index=False))
