import csv
import hashlib
import json
from pathlib import Path


def split_of(key):
    """Stable split; key must identify underlying IMAGE, not audio/caption."""
    bucket=int(hashlib.sha256(str(key).encode()).hexdigest()[:8],16)%100
    return 'train' if bucket<80 else 'val' if bucket<90 else 'test'


def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as out:
        for row in rows: out.write(json.dumps(row,ensure_ascii=False)+'\n')


def append_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf8') as out:
        for row in rows: out.write(json.dumps(row,ensure_ascii=False)+'\n')


def read_csv(path):
    with open(path,newline='',encoding='utf8') as f:
        return list(csv.DictReader(f))


def common_path(root,path):
    p=Path(path)
    return str(p if p.is_absolute() else Path(root)/p)


def safe_norm_box(obj,width,height):
    x,y,w,h=[float(obj.get(k,0)) for k in ('x','y','w','h')]
    return [max(0.,min(1.,x/width)),max(0.,min(1.,y/height)),
            max(0.,min(1.,(x+w)/width)),max(0.,min(1.,(y+h)/height))]
