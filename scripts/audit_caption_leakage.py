import argparse
import json
import re
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--report',type=Path,default=Path('results/caption_leakage.jsonl'))
a=p.parse_args()
a.report.parent.mkdir(parents=True,exist_ok=True)
total=0;flagged=0
with a.manifest.open(encoding='utf8') as stream,a.report.open('w',encoding='utf8') as out:
    for line in stream:
        if not line.strip(): continue
        rec=json.loads(line);total+=1
        targets=rec.get('labels',{}).get('categories',[])
        captions=[o.get('text','') for o in rec.get('observations',[])]
        detected=sorted({c for c in targets if any(re.search(r'(?<!\w)'+re.escape(c.lower())+r'(?!\w)', t.lower()) for t in captions)})
        if detected:
            flagged+=1
            out.write(json.dumps({'sample_id':rec.get('sample_id'),'named_target_categories':detected})+'\n')
print(f'Potential target-label mentions: {flagged}/{total}. Review these cases before reporting recognition gains.')
