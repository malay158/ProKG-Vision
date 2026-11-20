import argparse
import math
import random
import subprocess
import shutil
import sys
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import write_jsonl

COLOURS={'red':(219,68,73),'blue':(55,113,220),'green':(56,152,99),'yellow':(239,190,61)}
SHAPES=['square','circle','triangle']


def draw_shape(draw,shape,box,color):
    if shape=='square': draw.rectangle(box,fill=color)
    elif shape=='circle': draw.ellipse(box,fill=color)
    else:
        l,t,r,b=box;draw.polygon([(l+r)//2,t,r,b,l,b],fill=color)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--count',type=int,default=36)
    p.add_argument('--with-speech',action='store_true',
                   help='Use locally installed espeak to synthesize spoken examples')
    args=p.parse_args()
    root=args.root.resolve();random.seed(22)
    (root/'data/demo/images').mkdir(parents=True,exist_ok=True)
    (root/'data/demo/audio').mkdir(parents=True,exist_ok=True)
    items={'train':[],'val':[],'test':[]}
    can_speak=args.with_speech and shutil.which('espeak') is not None
    for i in range(args.count):
        obj1,obj2=SHAPES[i%3],SHAPES[(i+1)%3]
        color1,color2=list(COLOURS)[i%4],list(COLOURS)[(i+2)%4]
        row=i//12
        image=Image.new('RGB',(128,128),(242,243,245))
        draw=ImageDraw.Draw(image)
        box_a=(12,35+(i%3)*5,50,79+(i%3)*5)
        box_b=(76,25+(i%2)*15,119,65+(i%2)*15)
        draw_shape(draw,obj1,box_a,COLOURS[color1])
        draw_shape(draw,obj2,box_b,COLOURS[color2])
        image_file=f'data/demo/images/scene_{i:03}.png';image.save(root/image_file)
        norm=lambda b:[round(b[0]/128,5),round(b[1]/128,5),round(b[2]/128,5),round(b[3]/128,5)]
        supported=f'The {obj1} is {color1}.'
        alternate=next(c for c in COLOURS if c not in (color1,color2))
        contradicted=f'The {obj1} is {alternate}.'
        insufficient=f'The {obj1} was manufactured yesterday.'
        observations=[]
        for t,(sentence,evidence) in enumerate([(supported,'supported'),
                                                 (contradicted,'contradicted'),
                                                 (insufficient,'insufficient')]):
            audio=f'data/demo/audio/scene_{i:03}_{t}.wav' if can_speak else ''
            if can_speak:
                subprocess.run(['espeak','-s','140','-w',str(root/audio),sentence],
                               stdout=subprocess.DEVNULL,check=True)
            observations.append({'text':sentence,'audio':audio,
                                  'evidence':evidence,'target_box':norm(box_a),
                                  **({'conflicts_with':color1} if evidence=='contradicted' else {})})
        rec={'sample_id':f'demo:{i:03}','image_id':f'demo:{i:03}',
             'source_image_key':f'demo:{i:03}','source':'synthetic_demo',
             'image':image_file,'observations':observations,
             'objects':[{'id':'a','category':obj1,'bbox':norm(box_a),'attributes':[color1]},
                        {'id':'b','category':obj2,'bbox':norm(box_b),'attributes':[color2]}],
             'relations':[{'subject_id':'a','predicate':'left of','object_id':'b'}],
             'labels':{'attributes':sorted({color1,color2}),
                       'relations':['left of'],'categories':sorted({obj1,obj2})}}
        split='train' if i%6 not in (4,5) else ('val' if i%6==4 else 'test')
        items[split].append(rec)
    for name,records in items.items():
        write_jsonl(root/f'data/demo/{name}.jsonl',records)
    print({k:len(v) for k,v in items.items()})
    print('Speech generated:',can_speak)
    print('All data here are synthetic demonstrations, never benchmark data.')

if __name__=='__main__':main()
