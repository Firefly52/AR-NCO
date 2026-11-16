#!/usr/bin/env python3
"""Re-run the saved generalization LKH protocol with an externally installed LKH-3."""
import argparse,hashlib,json,re,subprocess,time
from pathlib import Path
import numpy as np
import torch
from lkh import export_validation_to_tsplib as exporter
from lkh import run_lkh_batch as interface
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--instances',type=int)
a=p.parse_args();binary=Path(a.binary).resolve();assert binary.is_file(),'Install LKH-3 externally first'
source=Path(a.dataset).resolve();out=Path(a.output).resolve()
for protected in ['results','datasets','checkpoints','code','configs']:assert not out.is_relative_to(ROOT/protected)
out.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'configs/generalization/protocol.json').read_text())['lkh']
for key in ['RUNS','MAX_TRIALS','TIME_LIMIT']:setattr(interface,key,cfg[key])
exporter.SCALE=cfg['coordinate_scale'];data=torch.load(source,weights_only=True,map_location='cpu')
count=len(data) if a.instances is None else a.instances;assert 0<count<=len(data)
meta={'dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'instances':count,'settings':cfg}
mp=out/'protocol.json'
if mp.exists():assert json.loads(mp.read_text())==meta
else:mp.write_text(json.dumps(meta,indent=2))
for i,points in enumerate(data[:count]):
 stem=out/f'instance_{i:06d}';result=stem.with_suffix('.json')
 if result.exists():assert json.loads(result.read_text())['status']=='success';continue
 tsp=stem.with_suffix('.tsp');par=stem.with_suffix('.par');tour=stem.with_suffix('.tour');log=stem.with_suffix('.log')
 if log.exists():raise RuntimeError(f'Inspect previous failed attempt before retrying: {log.name}')
 exporter._write_tsplib_file(points,tsp,stem.name)
 par.write_text(interface._build_par_content(tsp.name,stem.name+'.last.tour')+f'TOUR_FILE = {tour.name}\nSEED = {cfg["SEED"]}\n')
 started=time.perf_counter()
 with log.open('x') as f:subprocess.run([str(binary),par.name],cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=cfg['RUNS']*cfg['TIME_LIMIT']+180,check=True)
 route=[];active=False
 for line in tour.read_text().splitlines():
  if line.strip()=='TOUR_SECTION':active=True;continue
  if active:
   if line.strip() in ['-1','EOF']:break
   route.extend(int(v)-1 for v in line.split())
 assert sorted(route)==list(range(len(points)))
 xy=points.double().numpy()[route];length=float(np.linalg.norm(xy-np.roll(xy,-1,axis=0),axis=1).sum())
 result.write_text(json.dumps({'status':'success','instance_id':i,'length':length,'route':route,'seconds':time.perf_counter()-started,'dataset_sha256':meta['dataset_sha256']},indent=2))
print(f'Completed {count} LKH references')
