import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prokg.data import loader
from prokg.model import ProKGVision
from prokg.metrics import multilabel_macro_f1,multilabel_map

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('results/stagewise.csv'))
p.add_argument('--batch-size',type=int,default=8)
a=p.parse_args()
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];v=saved['vocab']
model=ProKGVision(*(len(v[k]) for k in ('objects','attributes','relations','categories')),
                  method=saved['method'],dim=cfg['dim'],grid=cfg['grid'],threshold=cfg['threshold'])
model.load_state_dict(saved['state_dict']);model.eval()
data=loader(a.manifest,v,batch_size=a.batch_size,image_size=cfg['image_size'],
            max_observations=cfg['max_observations'],max_audio_seconds=cfg['max_audio_seconds'],grid=cfg['grid'])
store={k:[] for k in ['attributes','relations','categories','stage_attributes','stage_relations','stage_categories',
                     'has_attr','has_rel','has_cat']}
with torch.no_grad():
    for batch in data:
        pred=model(batch['image'],batch['audio'],batch['tokens'],batch['obs_mask'])
        for k in store:
            store[k].append((pred[k] if k.startswith('stage_') else batch[k]).detach())
store={k:torch.cat(vals).numpy() for k,vals in store.items()}
rows=[]
for step in range(cfg['max_observations']+1):
    row={'stage':step,'source':'measured_model_outputs'}
    for task,label,logits,available,metric in [
        ('attribute_macro_F1_percent','attributes','stage_attributes','has_attr',multilabel_macro_f1),
        ('relation_macro_F1_percent','relations','stage_relations','has_rel',multilabel_macro_f1),
        ('multilabel_mAP_percent','categories','stage_categories','has_cat',multilabel_map)]:
        mask=store[available].astype(bool)
        row[task]=metric(store[label][mask],store[logits][mask,step]) if mask.any() else None
    rows.append(row)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Saved measured stage results:',a.output)
