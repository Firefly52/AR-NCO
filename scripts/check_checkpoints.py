#!/usr/bin/env python3
"""Strict-load every manifest entry; no key filling or layer substitution."""
import argparse,json
from pathlib import Path
import torch
from evaluate import manifest,load_model,state_hash
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(2);report=[]
for entry in manifest():
 model=load_model(entry,torch.device('cpu'))
 report.append({'id':entry['id'],'strict_load':True,'parameters':sum(x.numel() for x in model.parameters()),'state_sha256':state_hash(model)})
 print(entry['id'],'OK',flush=True)
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2))
