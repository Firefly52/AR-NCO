#!/usr/bin/env python3
"""Strict checkpoint evaluation using the retained AM/POMO model implementations."""
import argparse, csv, hashlib, importlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def manifest():
    return json.loads((ROOT / 'checkpoints/manifest.json').read_text())

def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_model(entry, device):
    path = ROOT / entry['file']
    assert sha(path) == entry['sha256'], 'Checkpoint SHA256 mismatch'
    cfg = json.loads((ROOT / entry['model_config']).read_text())
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if entry['solver'] == 'AM':
        sys.path.insert(0, str(ROOT / 'code/AM'))
        from problems import TSP, CVRP
        cls = importlib.import_module(entry['model_module']).AttentionModel
        model = cls(cfg['embedding_dim'], cfg['hidden_dim'], TSP if entry['problem']=='tsp' else CVRP,
                    n_encode_layers=cfg['n_encode_layers'], normalization=cfg['normalization'],
                    tanh_clipping=cfg['tanh_clipping'], mask_inner=True, mask_logits=True,
                    checkpoint_encoder=cfg.get('checkpoint_encoder', False), shrink_size=cfg.get('shrink_size'))
        state = ckpt['model']; model.set_decode_type('greedy')
    else:
        folder = ROOT / 'code/POMO_PO/POMO' / entry['problem'].upper()
        sys.path[:0] = [str(folder/'POMO'),str(folder)]
        cls = getattr(importlib.import_module(entry['model_module']),entry['problem'].upper()+'Model')
        model = cls(**{**cfg,'eval_type':'argmax'}); state = ckpt['model_state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

@torch.inference_mode()
def decode(model, entry, data, device):
    b = len(data); problem = entry['problem']; n = data.shape[1] - (problem=='cvrp')
    if entry['solver'] == 'AM':
        batch = data.to(device) if problem=='tsp' else {'depot':data[:,0,:2].to(device),'loc':data[:,1:,:2].to(device),'demand':data[:,1:,2].to(device)}
        lengths, _, routes = model(batch, return_pi=True)
        # The original problem.get_costs validates visits and capacity during forward.
        return lengths.cpu().numpy().astype(float), routes.cpu().numpy()
    cls = getattr(importlib.import_module(problem.upper()+'Env'),problem.upper()+'Env')
    env = cls(problem_size=n,pomo_size=n,device=device)
    if problem=='tsp':
        env.load_problems(b, problems=data.to(device), aug_factor=1)
    else:
        env.FLAG__use_saved_problems=True;env.saved_index=0
        env.saved_depot_xy=data[:,0:1,:2];env.saved_node_xy=data[:,1:,:2];env.saved_node_demand=data[:,1:,2]
        env.load_problems(b,aug_factor=1)
    reset,_,_=env.reset();state,reward,done=env.pre_step();model.pre_forward(reset)
    steps=0
    while not done:
        selected,_=model(state);state,reward,done=env.step(selected);steps+=1
        assert steps<=2*n+2,'Decoder failed to terminate'
    best,starts=reward.max(dim=1)
    all_routes=env.selected_node_list.cpu().numpy()
    if problem=='tsp':
        assert np.all(np.sort(all_routes,axis=-1)==np.arange(n)),'Invalid TSP tour'
    else:
        # Validate each POMO start, including capacity and customer uniqueness.
        for i in range(b):
            for route in all_routes[i]:
                assert np.array_equal(np.sort(route[route>0]),np.arange(1,n+1)),'Invalid customer visits'
                load=0.
                for city in route:
                    load=0. if city==0 else load+float(data[i,city,2])
                    assert load<=1+1e-5,'Capacity exceeded'
    routes=all_routes[np.arange(b),starts.cpu().numpy()]
    coords=data[:,:,:2].double().numpy();ordered=np.take_along_axis(coords,routes[:,:,None],axis=1)
    lengths=np.linalg.norm(ordered-np.roll(ordered,-1,axis=1),axis=2).sum(1)
    assert np.allclose(lengths,(-best).cpu().numpy(),rtol=2e-6,atol=2e-6)
    return lengths,routes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--instances',type=int);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--device',default='cpu');p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(2)
    entry=next(x for x in manifest() if x['id']==args.id)
    source=Path(args.dataset);source=source if source.is_absolute() else ROOT/source
    data=torch.load(source,map_location='cpu',weights_only=True)
    count=len(data) if args.instances is None else args.instances
    assert 0<count<=len(data) and args.batch_size>0
    if entry['problem']=='cvrp':assert data.shape[-1]==3
    else:assert data.shape[-1]==2
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    # Saved evidence is immutable. New evaluation outputs belong outside these trees.
    for name in ['results','datasets','checkpoints','code','configs']:
        assert not out.is_relative_to(ROOT/name),'Use a new output directory, e.g. outputs/evaluation'
    protocol={'checkpoint':entry['sha256'],'dataset':sha(source),'instances':count,'batch_size':args.batch_size,'device':args.device,'method':entry['id'],'decoding':'greedy multi-start, augmentation 1' if entry['solver']!='AM' else 'greedy, augmentation 1'}
    meta=out/'protocol.json'
    if meta.exists():assert json.loads(meta.read_text())==protocol,'Output belongs to another evaluation'
    else:meta.write_text(json.dumps(protocol,indent=2))
    device=torch.device(args.device);model=load_model(entry,device);before=state_hash(model);rows=[]
    for start in range(0,count,args.batch_size):
        path=out/f'batch_{start:06d}.npz';stop=min(start+args.batch_size,count)
        if path.exists():
            with np.load(path) as saved:
                assert saved['checkpoint'].item()==protocol['checkpoint'] and saved['dataset'].item()==protocol['dataset']
                assert np.array_equal(saved['instance_ids'],np.arange(start,stop))
                lengths=saved['lengths'];elapsed=float(saved['seconds'])
        else:
            if device.type=='cuda':torch.cuda.synchronize(device)
            t=time.perf_counter();lengths,routes=decode(model,entry,data[start:stop],device)
            if device.type=='cuda':torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-t
            tmp=path.with_suffix('.tmp.npz');np.savez(tmp,lengths=lengths,routes=routes,seconds=elapsed,instance_ids=np.arange(start,stop),checkpoint=protocol['checkpoint'],dataset=protocol['dataset']);tmp.replace(path)
        assert len(lengths)==stop-start and np.isfinite(lengths).all()
        rows.extend({'instance_id':start+i,'length':float(v)} for i,v in enumerate(lengths))
    assert before==state_hash(model),'Parameters changed'
    assert all(x.grad is None for x in model.parameters())
    with (out/'per_instance.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['instance_id','length']);w.writeheader();w.writerows(rows)
    summary={'method':entry['id'],'instances':len(rows),'mean_length':float(np.mean([x['length'] for x in rows])),'strict_load':True,'parameters_unchanged':True,'all_tours_validated':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
