import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='small')
    p.add_argument('--device',default='cpu')
    p.add_argument('--language',default='en')
    args=p.parse_args()
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit('pip install faster-whisper')
    whisper=WhisperModel(args.model,device=args.device,compute_type='int8' if args.device=='cpu' else 'float16')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.input.open(encoding='utf8') as source,args.output.open('w',encoding='utf8') as target:
        for line in source:
            if not line.strip():continue
            r=json.loads(line)
            for obs in r.get('observations',[]):
                audio=obs.get('audio','')
                if not audio:continue
                if 'reference_transcript' not in obs:
                    obs['reference_transcript']=obs.get('text','')
                segments,_=whisper.transcribe(audio,language=args.language)
                obs['text']=' '.join(s.text.strip() for s in segments)
                obs['transcript_source']='faster-whisper-asr'
                obs['asr_model']=args.model
            target.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('Wrote ASR transcripts:',args.output)

if __name__=='__main__': main()
