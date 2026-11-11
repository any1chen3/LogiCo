"""Compare fixed, full-image and masked normal reference retrieval on cached RGB proposals."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'saved_results/context_e3/cache_v1'
OUT=ROOT/'saved_results/context_e3/evaluation_v1'


def read(path):
    with np.load(path) as f: return {key:f[key] for key in f.files if key!='masks'}


def queries():
    result=[(f'small_{y}_{x}',[y*4+x],(x/4,y/4,(x+1)/4,(y+1)/4)) for y in range(4) for x in range(4)]
    result += [(f'large_{y}_{x}',[(2*y+dy)*4+2*x+dx for dy in range(2) for dx in range(2)],(x/2,y/2,(x+1)/2,(y+1)/2)) for y in range(2) for x in range(2)]
    return result


def indices(data,box):
    x1,y1,x2,y2=box; c=data['region_centers']
    return np.flatnonzero((c[:,0]>=x1)&(c[:,0]<x2)&(c[:,1]>=y1)&(c[:,1]<y2))


def error(expected, observed, box):
    a,b=indices(expected,box),indices(observed,box); m,n=len(a),len(b)
    if not m or not n: return .35*(m+n)/max(1,m+n)
    av,bv=expected['region_features'][a],observed['region_features'][b]
    appearance=np.clip((1-av@bv.T)/2,0,1)
    color=np.clip(np.linalg.norm(expected['region_lab'][a,None,:]-observed['region_lab'][None,b,:],axis=2)/100,0,1)
    pos=np.clip(np.linalg.norm(expected['region_centers'][a,None,:]-observed['region_centers'][None,b,:],axis=2)/np.sqrt(2),0,1)
    area=np.clip(np.abs(np.log(expected['region_area'][a,None]/observed['region_area'][None,b]))/np.log(4),0,1)
    real=.5*appearance+.2*color+.15*pos+.15*area
    cost=np.full((m+n,m+n),np.inf); cost[:m,:n]=real
    cost[np.arange(m),n+np.arange(m)]=.35; cost[m+np.arange(n),np.arange(n)]=.35; cost[m:,n:]=0
    r,c=linear_sum_assignment(cost)
    return float(cost[r,c].sum()/max(1,m+n))


def distance(query,bank,visible):
    f=query['tile_features'][visible]; labs=query['tile_lab'][visible]
    scores=[]
    for path,data in bank:
        visual=np.clip((1-(data['tile_features'][visible]*f).sum(axis=1))/2,0,1)
        color=np.clip(np.linalg.norm(data['tile_lab'][visible]-labs,axis=1)/100,0,1)
        scores.append((float(np.mean(.8*visual+.2*color)),path.name))
    return sorted(scores)


def main():
    OUT.mkdir(parents=True,exist_ok=True); allrows=[]; interventions=[]
    for category in ['splicing_connectors','juice_bottle']:
        train=sorted((CACHE/category/'train').glob('*.npz')); validation=sorted((CACHE/category/'validation').glob('*.npz'))
        expected_train=len(list((ROOT/'data/mvtec_loco'/category/'train/good').glob('*.png')))
        expected_val=len(list((ROOT/'data/mvtec_loco'/category/'validation/good').glob('*.png')))
        if len(train)!=expected_train or len(validation)!=expected_val:
            raise RuntimeError(f'{category} cache incomplete: train {len(train)}/{expected_train}, val {len(validation)}/{expected_val}')
        bank=[(p,read(p)) for p in train]; lookup={p.name:d for p,d in bank}
        mean_features=np.mean(np.stack([d['tile_features'] for _,d in bank]),axis=0)
        ranked=sorted((float(np.linalg.norm(d['tile_features']-mean_features)),p.name) for p,d in bank)
        fixed=[ranked[i][1] for i in np.linspace(0,len(ranked)-1,5,dtype=int)]
        # Lexicographic first half reserved for later calibration; second half
        # is the normal mechanism-check set, fixed before evaluation.
        vals=[(p,read(p)) for p in validation[len(validation)//2:]]
        for index,(path,source) in enumerate(vals):
            donorpath,donor=vals[(index+1)%len(vals)]
            full=[x[1] for x in distance(source,bank,list(range(16)))[:5]]
            for name,hidden,box in queries():
                visible=[i for i in range(16) if i not in hidden]
                masked=[x[1] for x in distance(source,bank,visible)[:5]]
                modes={'fixed':fixed,'full':full,'masked':masked}
                for mode,ids in modes.items():
                    errs=[error(lookup[id],source,box) for id in ids]
                    allrows.append(dict(category=category,image=path.stem,query=name,mode=mode,error=float(np.median(errs)),
                                        observed_count=len(indices(source,box)),reference_ids=';'.join(ids)))
                donor_ids=[x[1] for x in distance(donor,bank,visible)[:5]]
                before_src=np.median([error(lookup[id],source,box) for id in masked])
                before_donor=np.median([error(lookup[id],donor,box) for id in masked])
                after_src=np.median([error(lookup[id],source,box) for id in donor_ids])
                after_donor=np.median([error(lookup[id],donor,box) for id in donor_ids])
                interventions.append(dict(category=category,source=path.stem,donor=donorpath.stem,query=name,
                    original_ids=';'.join(masked),context_swapped_ids=';'.join(donor_ids),
                    before_source_error=float(before_src),before_donor_error=float(before_donor),
                    after_source_error=float(after_src),after_donor_error=float(after_donor),
                    shift_toward_donor=float((after_src-after_donor)-(before_src-before_donor)),
                    source_donor_distance=error(source,donor,box)))
            print(category,index+1,'/',len(vals),flush=True)
    for filename,rows in [('masked_prediction.csv',allrows),('interventions.csv',interventions)]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    summary={}
    for category in ['splicing_connectors','juice_bottle']:
        summary[category]={}
        for mode in ['fixed','full','masked']:
            rows=[r for r in allrows if r['category']==category and r['mode']==mode]
            image_means=[np.mean([r['error'] for r in rows if r['image']==image]) for image in sorted({r['image'] for r in rows})]
            summary[category][mode]=dict(images=len(image_means),queries=len(rows),mean_image_error=float(np.mean(image_means)),empty_observed=sum(r['observed_count']==0 for r in rows))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
