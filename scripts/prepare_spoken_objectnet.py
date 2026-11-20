import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import read_csv,write_jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/processed/son/test.jsonl'))
    args=p.parse_args()
    rows=[]
    for i,row in enumerate(read_csv(args.pairs)):
        key='son:'+str(row['image_id'])
        rows.append({'sample_id':f'{key}:{i}','image_id':key,'source_image_key':key,
                     'source':'spoken_objectnet','image':row['image_path'],
                     'observations':[{'audio':row['audio_path'],'text':row.get('transcript','')}],
                     'labels':{}})
    write_jsonl(args.output,rows)
    print(f'Wrote {len(rows)} held-out retrieval queries to {args.output}')
    print('Never train/fine-tune on these records for zero-shot target-domain evaluation.')

if __name__=='__main__':main()
