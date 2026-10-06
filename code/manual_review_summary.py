"""Reproduce the descriptive author-inspection counts in Table S21(b)."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
labels={"可辨认物体":"Recognizable object","物体局部或含糊区域":"Object part or ambiguous region","背景或纹理":"Background or texture","无法判断":"Unable to judge"}
with (ROOT/'source_data/manual_review_20261006/ratings_anonymous.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==150 and len({r['sample_id'] for r in rows})==150
assert all(r['category'] in labels and r['reviewer']=='Author' for r in rows)
summary=[dict(category=v,count=sum(r['category']==k for r in rows),percent_of_150=round(100*sum(r['category']==k for r in rows)/len(rows),1)) for k,v in labels.items()]
print(json.dumps(summary,indent=2))
