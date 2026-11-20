import argparse
import json
import random
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import append_jsonl,split_of,safe_norm_box

COLOURS=('red','blue','green','yellow','black','white','orange','purple')


@contextmanager
def stream_json(path):
    path=Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            name=next((n for n in z.namelist() if n.endswith('.json')),None)
            if not name: raise ValueError('No JSON in zip')
            with z.open(name) as f: yield f
    else:
        with path.open('rb') as f: yield f


def scenes(path,limit=None):
    try:
        import ijson
    except ImportError as ex:
        raise RuntimeError('pip install ijson to parse large Visual Genome JSON safely') from ex
    with stream_json(path) as f:
        for ix,row in enumerate(ijson.items(f,'item')):
            if limit and ix>=limit: break
            yield row


def convert(row,image_root,image_meta=None):
    image_id=str(row.get('image_id',row.get('id')))
    if image_id=='None': return None
    info=(image_meta or {}).get(image_id,{})
    width=int(row.get('width') or info.get('width') or 0)
    height=int(row.get('height') or info.get('height') or 0)
    if not width or not height: return None
    image=Path(image_root)/f'{image_id}.jpg'
    if not image.exists():
        candidates=[Path(image_root)/'VG_100K'/f'{image_id}.jpg',
                    Path(image_root)/'VG_100K_2'/f'{image_id}.jpg']
        image=next((p for p in candidates if p.exists()),image)
        if not image.exists(): return None
    objects=[]
    mapping={}
    for item in row.get('objects',[]):
        obj_id=str(item.get('object_id',item.get('id','')))
        names=item.get('names') or [item.get('name','object')]
        label=str(names[0]).strip().lower()
        attributes=[str(a).strip().lower() for a in item.get('attributes',[]) if str(a).strip()]
        box=safe_norm_box(item,width,height)
        objects.append({'id':obj_id,'category':label,'attributes':attributes,'bbox':box})
        mapping[obj_id]=objects[-1]
    relations=[]
    for rel in row.get('relationships',[]):
        sid=str(rel.get('subject_id') or (rel.get('subject') or {}).get('object_id',''))
        oid=str(rel.get('object_id') or (rel.get('object') or {}).get('object_id',''))
        if sid in mapping and oid in mapping:
            relations.append({'subject_id':sid,'predicate':str(rel.get('predicate','')).strip().lower(),
                              'object_id':oid})
    attributes=sorted(set(a for obj in objects for a in obj['attributes']))
    predicates=sorted(set(r['predicate'] for r in relations if r['predicate']))
    observations=[]
    eligible=next((o for o in objects if o['attributes']),None)
    if eligible:
        att=eligible['attributes'][0]
        observations.append({'text':f"The {eligible['category']} is {att}.",
                             'audio':'','evidence':'supported','target_box':eligible['bbox']})
        # A different annotated colour may ALSO be present on a multicoloured object.
        # Mark as unverified until a person confirms the image-level contradiction.
        att_color=next((a for a in eligible['attributes'] if a in COLOURS),None)
        if att_color:
            alternate=next(c for c in COLOURS if c!=att_color and c not in eligible['attributes'])
            observations.append({'text':f"The {eligible['category']} is {alternate}.",
                                 'audio':'','candidate_evidence':'contradicted',
                                 'needs_visual_verification':True,
                                 'target_box':eligible['bbox'],
                                 'conflicts_with':att_color})
        observations.append({'text':f"The {eligible['category']} was manufactured yesterday.",
                             'audio':'','evidence':'insufficient','target_box':eligible['bbox']})
    elif objects:
        eligible=objects[0]
        observations.append({'text':f"There is a {eligible['category']}.",
                             'audio':'','evidence':'supported','target_box':eligible['bbox']})
        observations.append({'text':f"The {eligible['category']} was manufactured yesterday.",
                             'audio':'','evidence':'insufficient','target_box':eligible['bbox']})
    image_key=str(info.get('coco_id') or f'vg:{image_id}')
    return {'sample_id':f'vg:{image_id}','image_id':f'vg:{image_id}',
            'source_image_key':f"coco:{image_key}" if info.get('coco_id') else f'vg:{image_id}',
            'source':'visual_genome','image':str(image),'observations':observations,
            'labels':{'attributes':attributes,'relations':predicates},
            'objects':objects,'relations':relations,'original_image_id':image_id}


def load_meta(path):
    if not path: return {}
    with stream_json(path) as f:
        return {str(x['image_id']):x for x in json.load(f)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene-graphs',type=Path,required=True)
    p.add_argument('--image-root',type=Path,required=True)
    p.add_argument('--image-metadata',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('data/processed/vg'))
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in ('train','val','test'):
        (args.output_dir/f'{name}.jsonl').write_text('')
    meta=load_meta(args.image_metadata)
    counts=Counter()
    for scene in scenes(args.scene_graphs,args.limit):
        record=convert(scene,args.image_root,meta)
        if record is None: counts['skipped_missing_image_or_size']+=1;continue
        split=split_of(record['source_image_key'])
        append_jsonl(args.output_dir/f'{split}.jsonl',[record]); counts[split]+=1
    print(dict(counts))
    print('NOTE: Audio paths are empty. Generate speech from verified annotations before training.')
    print('NOTE: Contradiction CANDIDATES are NOT evidence labels until visually checked.')

if __name__=='__main__': main()
