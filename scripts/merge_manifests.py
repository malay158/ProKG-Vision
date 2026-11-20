import argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--inputs',nargs='+',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('w',encoding='utf8') as out:
    for f in args.inputs:
        with f.open(encoding='utf8') as inp:
            for line in inp:
                if line.strip(): out.write(line.rstrip('\n')+'\n')
print('Created',args.output)
