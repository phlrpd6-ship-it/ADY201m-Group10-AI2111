from pathlib import Path
import pandas as pd, numpy as np, json
root=Path(__file__).resolve().parent.parent; out=Path(__file__).resolve().parent
combined=pd.read_csv(root/'cleaned'/'ADY201m_combined_cleaned.csv',low_memory=False)
details=pd.read_csv(out/'duplicate_record_details.csv',low_memory=False)
results=[]
for src,g in details.groupby('SOURCE_FILE',sort=True):
 raw=pd.read_excel(root/'raw'/src)
 for _,r in g.iterrows():
  q=raw[(raw['Mã học kỳ'].astype(str)==str(r['Mã học kỳ'])) & (raw.CODE.astype(str)==str(r.CODE))]
  if 'Mã lớp' in raw: q=q[q['Mã lớp'].astype(str)==str(r['Mã lớp'])]
  if 'NO.' in raw: q=q[q['NO.'].astype(str)==str(r['NO.'])]
  overlapping=[c for c in raw.columns if c in combined.columns]
  status='NO_RAW_MATCH'; mismatch=''; idx=''
  if len(q)==1:
   rr=q.iloc[0]; bad=[]
   for c in overlapping:
    a,b=rr[c],r[c]
    if pd.isna(a) and pd.isna(b): continue
    if pd.isna(a)!=pd.isna(b): bad.append(c); continue
    if isinstance(a,(int,float,np.number)) and isinstance(b,(int,float,np.number)):
     if not np.isclose(float(a),float(b),equal_nan=True): bad.append(c)
    elif str(a).strip()!=str(b).strip(): bad.append(c)
   status='UNIQUE_MATCH_VALUES_EQUAL' if not bad else 'UNIQUE_MATCH_VALUES_DIFFER'; mismatch=';'.join(bad); idx=str(q.index[0])
  elif len(q)>1: status='AMBIGUOUS_RAW_MATCH'
  results.append({'SOURCE_FILE':src,'CODE':r.CODE,'Mã học kỳ':r['Mã học kỳ'],'Mã lớp':r['Mã lớp'],'NO.':r['NO.'],'ROW_IDX_ORIG':r.ROW_IDX_ORIG,'candidate_count':len(q),'raw_candidate_indices':';'.join(map(str,q.index)),'status':status,'matched_raw_index':idx,'mismatch_columns':mismatch,'raw_candidates_json':' || '.join(json.dumps(row.to_dict(),ensure_ascii=False,default=str) for _,row in q.iterrows())})
df=pd.DataFrame(results); df.to_csv(out/'source_key_match.csv',index=False,encoding='utf-8-sig')
print(df.groupby(['SOURCE_FILE','status']).size().to_string())
