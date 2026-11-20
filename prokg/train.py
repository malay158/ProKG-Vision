import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from .data import loader,load_vocab
from .losses import prokg_loss
from .model import ProKGVision,METHODS


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run(args):
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    vocab=load_vocab(args.vocab)
    opts={'image_size':args.image_size,'max_observations':args.max_observations,
          'max_audio_seconds':args.max_audio_seconds,'grid':args.grid}
    training=loader(args.train,args.vocab,batch_size=args.batch_size,shuffle=True,**opts)
    validation=loader(args.val,args.vocab,batch_size=args.batch_size,**opts) if args.val else None
    model=ProKGVision(*(len(vocab[k]) for k in ('objects','attributes','relations','categories')),
                      method=args.method,dim=args.dim,grid=args.grid,threshold=args.threshold).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    best=float('inf')
    for epoch in range(1,args.epochs+1):
        model.train()
        losses=[]
        for batch in training:
            batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs=model(batch['image'],batch['audio'],batch['tokens'],batch['obs_mask'])
            loss,detail=prokg_loss(outputs,batch,args.method)
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        val_losses=[]
        if validation:
            with torch.no_grad():
                for batch in validation:
                    batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                    pred=model(batch['image'],batch['audio'],batch['tokens'],batch['obs_mask'])
                    loss,_=prokg_loss(pred,batch,args.method)
                    val_losses.append(loss.item())
        train_avg=float(np.mean(losses)); val_avg=float(np.mean(val_losses)) if val_losses else train_avg
        print(f'epoch={epoch} train_loss={train_avg:.4f} val_loss={val_avg:.4f}')
        if val_avg < best:
            best=val_avg
            torch.save({'state_dict':model.state_dict(),'method':args.method,
                        'vocab':vocab,'config':{'dim':args.dim,'grid':args.grid,
                        'threshold':args.threshold,'max_observations':args.max_observations,
                        'image_size':args.image_size,'max_audio_seconds':args.max_audio_seconds,'grid':args.grid},
                        'seed':args.seed,'epoch':epoch},args.output)
    print('Checkpoint saved:',args.output)


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--val',type=Path)
    p.add_argument('--vocab',type=Path,required=True)
    p.add_argument('--method',choices=METHODS,default='full')
    p.add_argument('--output',type=Path,default=Path('checkpoints/full.pt'))
    p.add_argument('--device',default='auto')
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--image-size',type=int,default=128)
    p.add_argument('--max-observations',type=int,default=4)
    p.add_argument('--max-audio-seconds',type=float,default=2.0)
    p.add_argument('--dim',type=int,default=64)
    p.add_argument('--grid',type=int,default=4)
    p.add_argument('--threshold',type=float,default=0.20)
    p.add_argument('--lr',type=float,default=2e-3)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    return p

if __name__=='__main__': run(arguments().parse_args())
