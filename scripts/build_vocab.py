import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,default=Path('configs/vocab_generated.json'))
    p.add_argument('--max-objects',type=int,default=200)
    p.add_argument('--max-attributes',type=int,default=100)
    p.add_argument('--max-relations',type=int,default=100)
    p.add_argument('--max-categories',type=int,default=80)
    a={k:Counter() for k in ('objects','attributes','relations','categories')}
    for filename in args.train:
        for line in filename.open(encoding='utf8'):
            if not line.strip(): continue
            r=json.loads(line)
            for o in r.get('objects',[]): a['objects'][o['category']]+=1
            for name in ('attributes','relations','categories'):
                a[name].update(set(r.get('labels',{}).get(name,[])))
    limits=dict(objects=args.max_objects,attributes=args.max_attributes,
                relations=args.max_relations,categories=args.max_categories)
    output={name:[label for label,_ in count.most_common(limits[name])]
            for name,count in a.items()}
    for name in output:
        if not output[name]: output[name]=['__none__']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print({k:len(v) for k,v in output.items()})

if __name__=='__main__':main()
