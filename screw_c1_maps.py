"""Reconstruct A/B and apply validation-only C1 to saved test branch outputs."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.logico import LogiCo

ROOT=Path(__file__).resolve().parent
AB=ROOT/'saved_results/ab_screw_bag'
OUT=ROOT/'saved_results/screw_c1'
CKPT=ROOT/'saved_results/mvtec_loco/screw_bag/LogiCo_best_ckpt.pkl'

def main(mode):
    calibration=json.loads((OUT/'calibration.json').read_text())
    original=json.loads((AB/'inference.json').read_text())
    with CKPT.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if calibration['status']!='valid' or sha!=calibration['checkpoint_sha256'] or sha!=original['checkpoint_sha256']:raise RuntimeError('Checkpoint or calibration mismatch')
    if mode=='apply':
        checked=json.loads((OUT/'ab_verification.json').read_text())
        if not checked.get('spro_passed'):raise RuntimeError('A/B sPRO must be verified before C1')
    with (AB/'scores.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=original['count'] or len(rows)!=341:raise RuntimeError('Incomplete test scores')
    ck=torch.load(CKPT,map_location='cpu',weights_only=True)
    device='cuda:0'
    stats={k:torch.as_tensor(ck[k],device=device) for k in ('CRN_mean','CRN_std','SRN_mean','SRN_std','SMD_mean','SMD_std')}
    del ck
    kernel=LogiCo.get_gaussian_kernel(None).to(device).eval()
    (OUT/'maps').mkdir(exist_ok=True)
    max_errors={'A':0.0,'B':0.0};new_rows=[]
    with torch.no_grad():
        for start in tqdm(range(0,len(rows),2),desc=mode):
            batch=rows[start:start+2];stored=[]
            for row in batch:
                with np.load(row['map_path']) as f:stored.append({k:f[k] for k in ('CRN','SRN','SMD','A','B')})
            def normalized(name):
                x=torch.from_numpy(np.stack([s[name] for s in stored])[:,None]).to(device)
                return (F.interpolate(x,size=(512,512),mode='bilinear',align_corners=True)-stats[name+'_mean'])/stats[name+'_std']
            crn=normalized('CRN');srn=normalized('SRN');smd=normalized('SMD');base=torch.maximum(crn,srn)
            def final(x):return F.interpolate(kernel(x),size=256,mode='bilinear',align_corners=False).cpu().numpy()[:,0]
            maps={'A':final(torch.maximum(base,smd)),'B':final(base)}
            for v in ('A','B'):
                error=float(np.max(np.abs(maps[v]-np.stack([s[v] for s in stored]))))
                max_errors[v]=max(max_errors[v],error)
                if error>1e-5:raise RuntimeError(f'{v} mismatch: {error}')
            if mode=='apply':
                c1=calibration['r95']+calibration['scale']*(smd-calibration['s95'])
                maps['C1']=final(torch.maximum(base,c1))
            for i,row in enumerate(batch):
                rel=Path(row['image_path']).relative_to(ROOT/'data/mvtec_loco/screw_bag/test')
                destination=OUT/'maps'/rel.parent/(rel.stem+'.npz');destination.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(destination,**{v:x[i] for v,x in maps.items()})
                new_rows.append({**row,'map_path':str(destination)})
    with (OUT/'scores.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=new_rows[0].keys());writer.writeheader();writer.writerows(new_rows)
    details={'mode':mode,'count':len(rows),'batch_size':2,'checkpoint_sha256':sha,'max_abs_difference_from_original':max_errors}
    (OUT/('reconstruction_'+mode+'.json')).write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(json.dumps(details,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('verify','apply'));main(parser.parse_args().mode)
