import argparse
import json
from pathlib import Path
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prokg.data import ProKGDataset
from prokg.model import ProKGVision

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--index',type=int,default=0)
p.add_argument('--output',type=Path,default=Path('results/qualitative_example.json'))
a=p.parse_args()
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];v=saved['vocab']
model=ProKGVision(*(len(v[k]) for k in ('objects','attributes','relations','categories')),
                  method=saved['method'],dim=cfg['dim'],grid=cfg['grid'],threshold=cfg['threshold'])
model.load_state_dict(saved['state_dict']);model.eval()
source=ProKGDataset(a.manifest,v,image_size=cfg['image_size'],max_observations=cfg['max_observations'],
                    max_audio_seconds=cfg['max_audio_seconds'],grid=cfg['grid'])
sample=source[a.index]
with torch.no_grad():
    output=model(*[sample[k][None] for k in ('image','audio','tokens','obs_mask')])
raw=source.items[a.index]
n=min(len(raw.get('observations',[])),cfg['max_observations'])
evidence_probs=output['evidence'][0].softmax(-1).tolist()
steps=[]
for step in range(n):
    ob=raw['observations'][step]
    steps.append({'step':step+1,'text':ob.get('text',''),
                  'reference_evidence':ob.get('evidence','NOT_ANNOTATED'),
                  'predicted_evidence_probabilities':dict(zip(
                      ['supported','contradicted','insufficient'],evidence_probs[step])),
                  'reliability_gate':float(output['gates'][0,step]),
                  'selected_region_grid_indices':(output['region_masks'][0,step]>0).nonzero().flatten().tolist()})
result={'image':raw['image'],'image_id':raw['image_id'],'model':saved['method'],
        'grid':cfg['grid'],'observations':steps,'note':'Actual checkpoint predictions, not an illustration.'}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print('Saved actual model predictions:',a.output)
