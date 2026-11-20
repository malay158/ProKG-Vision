import argparse
import subprocess
import sys
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prokg.model import METHODS

p=argparse.ArgumentParser()
p.add_argument('--train',type=Path,required=True)
p.add_argument('--val',type=Path,required=True)
p.add_argument('--test',type=Path,required=True)
p.add_argument('--vocab',type=Path,required=True)
p.add_argument('--epochs',type=int,default=5)
p.add_argument('--batch-size',type=int,default=8)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--dim',type=int,default=64)
a=p.parse_args()
for method in METHODS:
    checkpoint=Path('checkpoints')/f'{method}_seed{a.seed}.pt'
    output=Path('results')/f'{method}_seed{a.seed}.json'
    subprocess.run([sys.executable,'-m','prokg.train','--method',method,'--train',str(a.train),
                    '--val',str(a.val),'--vocab',str(a.vocab),'--epochs',str(a.epochs),
                    '--batch-size',str(a.batch_size),'--dim',str(a.dim),
                    '--seed',str(a.seed),'--output',str(checkpoint)],check=True)
    subprocess.run([sys.executable,'-m','prokg.evaluate','--checkpoint',str(checkpoint),
                    '--manifest',str(a.test),'--output',str(output)],check=True)
print('Finished six models. Do not compare task metrics across different datasets.')
