import argparse
import json
import sys
from collections import defaultdict,Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import read_csv,write_jsonl,split_of


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--instances',type=Path,required=True,
                   help='MS COCO instances_train2017.json (choose matching year/split)')
    p.add_argument('--output-dir',type=Path,default=Path('data/processed/spokencoco'))
    args=p.parse_args()
    coco=json.loads(args.instances.read_text())
    cats={int(c['id']):str(c['name']) for c in coco['categories']}
    labels=defaultdict(set)
    for a in coco['annotations']: labels[int(a['image_id'])].add(cats[int(a['category_id'])])
    grouped=defaultdict(list)
    for number,row in enumerate(read_csv(args.pairs)):
        image_id=int(row['image_id'])
        if image_id not in labels: continue
        split=(row.get('split') or '').lower() or split_of('coco:'+str(image_id))
        if split not in ('train','val','test'): raise ValueError(f'Invalid split: {split}')
        # All captions belonging to the same source image must share a split.
        grouped[split].append({'sample_id':f'coco:{image_id}:{number}',
              'image_id':f'coco:{image_id}', 'source_image_key':f'coco:{image_id}',
              'source':'spokencoco','image':row['image_path'],
              'observations':[{'audio':row['audio_path'],'text':row.get('transcript','')}],
              'labels':{'categories':sorted(labels[image_id])}})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for split in ('train','val','test'):
        write_jsonl(args.output_dir/f'{split}.jsonl',grouped[split])
    print({s:len(rows) for s,rows in grouped.items()})
    print('NOTE: Recorded speech captions are not verified or contradictory evidence labels.')
    print('NOTE: Use source release year and annotation version consistently; audit split/overlap.')

if __name__=='__main__':main()
