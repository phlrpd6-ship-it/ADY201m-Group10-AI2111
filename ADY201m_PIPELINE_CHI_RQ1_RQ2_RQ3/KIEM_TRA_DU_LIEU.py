# Kiem tra nhanh du lieu RQ1-RQ3, KHONG train
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parent
path=root/'DU_LIEU'/'ADY201m_combined_cleaned.csv'
if not path.exists():
    raise SystemExit(f'Thieu CSV: {path}')
df=pd.read_csv(path, low_memory=False)
need={'CODE','SOURCE_FILE','TARGET','HAS_PREFINAL'}
if not need <= set(df.columns):
    raise SystemExit(f'Thieu cot: {sorted(need - set(df.columns))}')
model=df[(df.HAS_PREFINAL == True) & df.TARGET.isin([0,1])]
print('CSV:',path.name)
print('Rows:',len(df),'| Sources:',df.SOURCE_FILE.nunique(),'| Columns:',len(df.columns))
print('Modeling rows:',len(model),'| Target counts:',model.TARGET.value_counts().to_dict())
print('Group keys:',model[['SOURCE_FILE','CODE']].drop_duplicates().shape[0])
print('OK - Chua huan luyen model')
