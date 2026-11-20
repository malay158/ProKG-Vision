import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from .data import loader
from .metrics import multilabel_macro_f1,multilabel_map,recall_at_k
from .model import ProKGVision


def evaluate(args):
    device=torch.device(args.device if args.device!='auto' else
                        ('cuda' if torch.cuda.is_available() else 'cpu'))
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    v=saved['vocab'];cfg=saved['config']
    model=ProKGVision(*(len(v[k]) for k in ('objects','attributes','relations','categories')),
                      method=saved['method'],dim=cfg['dim'],grid=cfg['grid'],
                      threshold=cfg['threshold']).to(device)
    model.load_state_dict(saved['state_dict']);model.eval()
    batches=loader(args.manifest,v,batch_size=args.batch_size,shuffle=False,
                   image_size=cfg['image_size'],max_observations=cfg['max_observations'],
                   max_audio_seconds=cfg['max_audio_seconds'],grid=cfg['grid'])
    outputs={'attr_y':[],'attr_s':[],'rel_y':[],'rel_s':[],'cat_y':[],
             'cat_s':[],'img':[],'aud':[],'ids':[]}
    with torch.no_grad():
        for batch in batches:
            pred=model(*[batch[k].to(device) for k in ('image','audio','tokens','obs_mask')])
            for target,head,flag in [('attr','attributes','has_attr'),('rel','relations','has_rel'),
                                     ('cat','categories','has_cat')]:
                usable=batch[flag].bool()
                outputs[f'{target}_y'].extend(batch[head][usable].tolist())
                outputs[f'{target}_s'].extend(pred[head].cpu()[usable].tolist())
            have=batch['has_retrieval'].bool()
            outputs['img'].extend(pred['image_embedding'].cpu()[have].tolist())
            outputs['aud'].extend(pred['speech_embedding'].cpu()[have].tolist())
            outputs['ids'].extend([k for k,b in zip(batch['image_id'],have) if b])
    metrics={'checkpoint_method':saved['method'],'num_records':len(batches.dataset),
             'sources':sorted({r.get('source','unknown') for r in batches.dataset.items}),
             'manifest':str(args.manifest),'checkpoint':str(args.checkpoint)}
    if outputs['attr_y']: metrics['attribute_macro_F1_percent']=multilabel_macro_f1(outputs['attr_y'],outputs['attr_s'])
    if outputs['rel_y']: metrics['relation_macro_F1_percent']=multilabel_macro_f1(outputs['rel_y'],outputs['rel_s'])
    if outputs['cat_y']: metrics['multilabel_mAP_percent']=multilabel_map(outputs['cat_y'],outputs['cat_s'])
    unique_images=len(set(outputs['ids']))
    if outputs['img'] and saved['method'] != 'image_only':
        if unique_images<10:
            warnings.warn(f'Only {unique_images} unique gallery images; R@10 is trivial and NOT publication-grade.')
            metrics['retrieval_small_gallery_warning']=True
            metrics['speech_image_Recall_at_10_percent']=None  # Suppress trivial 5/5 galleries.
        else:
            metrics['speech_image_Recall_at_10_percent']=recall_at_k(outputs['img'],outputs['aud'],outputs['ids'],k=10)
        metrics['retrieval_gallery_size']=unique_images
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf8')
    print(json.dumps(metrics,indent=2))
    return metrics

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/metrics.json'))
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='auto')
    evaluate(p.parse_args())
