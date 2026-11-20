import argparse
import json
import subprocess
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audio-dir',type=Path,required=True)
    p.add_argument('--speed',type=int,default=145)
    p.add_argument('--voice',default='en')
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if not shutil.which('espeak'): raise SystemExit('Install espeak or use your TTS engine to populate audio paths.')
    args.audio_dir.mkdir(parents=True,exist_ok=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with args.input.open(encoding='utf8') as source,args.output.open('w',encoding='utf8') as out:
        for i,line in enumerate(source):
            if args.limit and i>=args.limit: break
            if not line.strip():continue
            item=json.loads(line)
            for j,obs in enumerate(item.get('observations',[])):
                text=obs.get('text','').strip()
                if not text:continue
                dest=args.audio_dir/f'{i:07}_{j:02}.wav'
                subprocess.run(['espeak','-v',args.voice,'-s',str(args.speed),
                                '-w',str(dest),text],check=True,
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                obs['audio']=str(dest)
                obs['tts_engine']='espeak'
            out.write(json.dumps(item,ensure_ascii=False)+'\n');count+=1
    print('Converted',count,'records. Review audio quality before scientific evaluation.')

if __name__=='__main__':main()
