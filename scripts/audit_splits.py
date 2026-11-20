import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',nargs='*',type=Path,default=[])
    p.add_argument('--val',nargs='*',type=Path,default=[])
    p.add_argument('--test',nargs='*',type=Path,default=[])
    args=p.parse_args()
    by_split=defaultdict(set)
    for split in ('train','val','test'):
        for filename in getattr(args,split):
            with filename.open(encoding='utf8') as stream:
                for line in stream:
                    if not line.strip(): continue
                    obj=json.loads(line)
                    by_split[split].add(str(obj.get('source_image_key') or obj['image_id']))
    errors=[]
    for a,b in [('train','val'),('train','test'),('val','test')]:
        intersection=by_split[a]&by_split[b]
        if intersection:
            errors.append(f'{a}/{b}: {len(intersection)} shared images (e.g. {sorted(intersection)[:5]})')
    for s in ('train','val','test'): print(f'{s}: {len(by_split[s])} unique images')
    if errors: raise SystemExit('LEAKAGE DETECTED:\n'+'\n'.join(errors))
    print('PASS: no matching source image keys across supplied split manifests')

if __name__=='__main__':main()
