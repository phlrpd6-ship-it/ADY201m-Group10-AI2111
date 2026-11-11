from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parent.parent
raw=pd.read_excel(root/'raw'/'161810CJ24_TongHop.xlsx')
clean=pd.read_csv(root/'cleaned'/'161810CJ24_cleaned.csv',low_memory=False)
dups=pd.read_csv(Path(__file__).resolve().parent/'duplicate_record_details.csv')
dups=dups[dups.SOURCE_FILE=='161810CJ24_TongHop.xlsx']
checks=[]
for _,r in dups.iterrows():
    cand=raw[(raw['Mã học kỳ'].astype(str)==str(r['Mã học kỳ'])) & (raw.CODE.astype(str)==str(r.CODE))]
    checks.append({'CODE':r.CODE,'semester':r['Mã học kỳ'],'clean_row_idx':int(r.ROW_IDX_ORIG),'candidate_count':len(cand),'raw_indices':';'.join(map(str,cand.index)),'candidate_values_equal':None if len(cand)!=1 else bool(cand.iloc[0].equals(raw.iloc[cand.index[0]]))})
out=pd.DataFrame(checks)
out.to_csv(Path(__file__).resolve().parent/'cj24_raw_key_check.csv',index=False,encoding='utf-8-sig')
print('raw rows',len(raw),'clean rows',len(clean),'exact code-order matches',(raw.CODE.astype(str).values[:len(clean)]==clean.CODE.astype(str).values).sum())
print(out.candidate_count.value_counts().to_dict())
