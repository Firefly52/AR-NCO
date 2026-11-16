#!/usr/bin/env python3
"""Verify saved data/checkpoints and cross-check the recorded generalization results."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 checked=0
 for name in ['datasets/manifest.json','checkpoints/manifest.json']:
  for item in json.loads((ROOT/name).read_text()):
   file=ROOT/item['file'];assert file.is_file(),f'Missing artifact: {item["file"]}; extract all three archives at the same level'
   assert sha(file)==item['sha256'],f'Hash mismatch: {item["file"]}';checked+=1
   if name.startswith('datasets'):
    x=torch.load(file,map_location='cpu',weights_only=True);assert list(x.shape)==item['shape']
 refs={};valid_tours=0
 for n in [100,200,500]:
  data=torch.load(ROOT/f'datasets/generalization/formal/tsp{n}.pt',weights_only=True).double().numpy()
  rr=json.loads((ROOT/f'datasets/references/generalization/tsp{n}.json').read_text());assert len(rr)==len(data)==1000
  assert sorted(x['instance_id'] for x in rr)==list(range(1000))
  for row in rr:
   assert row['status']=='success' and row['returncode']==0
   route=np.asarray(row['route']);assert np.array_equal(np.sort(route),np.arange(n))
   xy=data[row['instance_id'],route];length=np.linalg.norm(xy-np.roll(xy,-1,axis=0),axis=1).sum()
   assert abs(length-row['length'])<1e-10
   refs[n,row['instance_id']]=row['length'];valid_tours+=1
 rows=list(csv.DictReader((ROOT/'results/generalization/all_per_instance.csv').open()));assert len(rows)==9000
 seen=set();methods={}
 for row in rows:
  n=int(row['size']);i=int(row['instance_id']);key=(row['method'],n,i);assert key not in seen;seen.add(key)
  ref=refs[n,i];length=float(row['neural_length']);gap=(length-ref)/ref*100
  assert abs(ref-float(row['lkh_length']))<1e-10 and abs(gap-float(row['relative_gap_percent']))<1e-10
  methods.setdefault((row['method'],n),[]).append(length)
 assert len(methods)==9 and all(len(v)==1000 for v in methods.values())
 for n in [100,200,500]:
  data=torch.load(ROOT/f'datasets/generalization/formal/tsp{n}.pt',weights_only=True).double().numpy()
  for method,label in [('baseline','POMO Baseline'),('post_ar','POMO + Post AR'),('full_ar','POMO + Full AR')]:
   collected={}
   for f in sorted((ROOT/f'results/generalization/tsp{n}/{method}').glob('batch_*.npz')):
    with np.load(f) as z:
     # Original experiment files identify their starting index in the filename.
     start=int(f.stem.split('_')[1]);routes=z['route'];lengths=z['length']
     assert np.array_equal(z['instance_id'],np.arange(start,start+len(routes)))
     assert np.all(np.sort(routes,axis=-1)==np.arange(n))
     xy=np.take_along_axis(data[start:start+len(routes)],routes[:,:,None],axis=1)
     actual=np.linalg.norm(xy-np.roll(xy,-1,axis=1),axis=2).sum(1)
     assert np.allclose(actual,lengths,rtol=0,atol=1e-10)
     for j,v in enumerate(actual):assert start+j not in collected;collected[start+j]=v
     valid_tours+=len(routes)
   assert len(collected)==1000
   relevant={int(x['instance_id']):float(x['neural_length']) for x in rows if x['method']==label and int(x['size'])==n}
   assert all(abs(collected[i]-relevant[i])<1e-10 for i in collected)
 report={'dataset_and_checkpoint_hashes':checked,'generalization_rows':len(rows),'validated_saved_tours':valid_tours,'passed':True}
 output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
