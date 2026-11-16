#!/usr/bin/env python3
"""Evaluate the recorded TSP100 epoch-500 policies on the saved test instances."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--sizes',type=int,nargs='+',choices=[100,200,500],default=[100,200,500])
p.add_argument('--instances',type=int,default=1000);p.add_argument('--batch-size',type=int,default=1)
p.add_argument('--device',default='cpu');p.add_argument('--output',required=True)
a=p.parse_args()
models=json.loads((ROOT/'configs/generalization/models.json').read_text())
for n in a.sizes:
 for method,ident in models.items():
  subprocess.run([sys.executable,str(ROOT/'scripts/evaluate.py'),'--id',ident,'--dataset',f'datasets/generalization/formal/tsp{n}.pt','--instances',str(a.instances),'--batch-size',str(a.batch_size),'--device',a.device,'--output',str(Path(a.output)/f'tsp{n}'/method)],check=True,cwd=ROOT)
