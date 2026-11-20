import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('files',nargs='+',type=Path)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--require-audio',action='store_true')
    args=p.parse_args()
    counts={};issues=[]
    for filepath in args.files:
        count=0
        for lineno,line in enumerate(filepath.open(encoding='utf8'),start=1):
            if not line.strip(): continue
            count+=1;record=json.loads(line)
            if not record.get('image_id'): issues.append(f'{filepath}:{lineno}: missing image_id')
            img=Path(record['image']);img=img if img.is_absolute() else args.root/img
            if not img.exists(): issues.append(f'{filepath}:{lineno}: missing {img}')
            for ob in record.get('observations',[]):
                if ob.get('evidence') and ob['evidence'] not in ('supported','contradicted','insufficient'):
                    issues.append(f'{filepath}:{lineno}: invalid evidence state')
                if args.require_audio and not ob.get('audio'):
                    issues.append(f'{filepath}:{lineno}: missing audio field')
                if ob.get('audio'):
                    aud=Path(ob['audio']);aud=aud if aud.is_absolute() else args.root/aud
                    if not aud.exists():issues.append(f'{filepath}:{lineno}: missing {aud}')
        counts[str(filepath)]=count
    print(counts)
    if issues: raise SystemExit('\n'.join(issues[:30]))
    print('PASS: manifest schema/path checks')

if __name__=='__main__':main()
