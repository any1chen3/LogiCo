"""Explore same empirical normal count/configuration rule on two LOCO classes."""
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'saved_results/context_e4_shared_normal_configuration'
CATEGORIES=('splicing_connectors','breakfast_box')
GRID=4

def features(path,nclasses):
 with Image.open(path) as im:
  if im.mode not in ('P','L'):raise ValueError('Not an integer label image: '+str(path))
  labels=np.asarray(im.resize((256,256),Image.Resampling.NEAREST))
 if labels.max()>=nclasses:raise ValueError('Unknown class label: '+str(path))
 counts=[]; occupancy=[]
 for cls in range(nclasses-1):
  mask=(labels==cls).astype('uint8')
  n,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
  counts.append(sum(stats[i,4]>=4 for i in range(1,n)))
  occupancy.extend([float(mask[y*64:(y+1)*64,x*64:(x+1)*64].mean()) for y in range(GRID) for x in range(GRID)])
 return np.asarray(counts,dtype=float),np.asarray(occupancy,dtype=float)

def scores(c,o,bank_c,bank_o):
 # Normal configuration is one complete training example, avoiding mixing categories across examples.
 count=np.maximum(bank_c-c,0).sum(axis=1)/np.maximum(bank_c.sum(axis=1),1)
 space=np.maximum(bank_o-o,0).sum(axis=1)/np.maximum(bank_o.sum(axis=1),1e-12)
 return .5*(count+space),count,space

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 allout={}
 for cat in CATEGORIES:
  seg=ROOT/'data/mvtec_loco_segmentations'/cat
  names=yaml.safe_load((ROOT/'configs/segmentations/mvtec_loco'/f'{cat}.yaml').read_text(encoding='utf-8'))['prompts']
  if names[-1].lower() not in ('black background','grid'):raise ValueError('Last label is not known background')
  paths=sorted((seg/'train/good').glob('*.png'))
  train=[features(p,len(names)) for p in paths];bank_c=np.stack([x[0] for x in train]);bank_o=np.stack([x[1] for x in train])
  loo=[]
  for i,(c,o) in enumerate(train):
   d,_,_=scores(c,o,bank_c,bank_o);d[i]=np.inf;loo.append(float(d.min()))
  cutoff=float(np.quantile(loo,.99,method='higher'))
  defects=defaultdict(set)
  with (ROOT/'saved_results'/('breakfast_ab' if cat=='breakfast_box' else 'ab_splicing_connectors')/'diagnosis/region_differences.csv').open(encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):defects[r['image']].add(r['defect_name'])
  rows=[]
  for split in ('validation/good','test/good','test/logical_anomalies'):
   for p in sorted((seg/split).glob('*.png')):
    c,o=features(p,len(names));d,dc,ds=scores(c,o,bank_c,bank_o);idx=int(np.argmin(d));labels=defects[p.name] if split=='test/logical_anomalies' else set()
    rows.append(dict(category=cat,split=split,image=p.name,defect=';'.join(sorted(labels)),score=float(d[idx]),count_deficit=float(dc[idx]),spatial_deficit=float(ds[idx]),flagged=bool(d[idx]>cutoff),nearest_training_image=paths[idx].name,observed_counts=';'.join(map(str,c.astype(int))),reference_counts=';'.join(map(str,bank_c[idx].astype(int)))))
  folder=OUT/cat;folder.mkdir(exist_ok=True)
  with (folder/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
  np.savez_compressed(folder/'normal_bank.npz',counts=bank_c,spatial_occupancy=bank_o,filenames=np.asarray([p.name for p in paths]),loo_scores=loo)
  distribution={names[i]:dict(sorted(Counter(str(int(x)) for x in bank_c[:,i]).items())) for i in range(len(names)-1)}
  def summary(r):return dict(images=len(r),flagged=sum(x['flagged'] for x in r),mean_score=float(np.mean([x['score'] for x in r])) if r else None)
  groups={s:summary([r for r in rows if r['split']==s]) for s in ('validation/good','test/good','test/logical_anomalies')}
  if cat=='breakfast_box':
   types={'0_nectarines_1_tangerine','0_nectarines_2_tangerines','1_nectarine_1_tangerine','0_nectarines_0_tangerines'}
   subset=[r for r in rows if r['split']=='test/logical_anomalies' and set(r['defect'].split(';'))&types]
  else:subset=[r for r in rows if r['split']=='test/logical_anomalies' and 'missing_cable' in r['defect'].split(';')]
  allout[cat]=dict(normal_training_images=len(paths),foreground_labels=names[:-1],training_component_counts=distribution,loo_threshold=cutoff,groups=groups,target_subset=summary(subset),target_ids=[r['image'] for r in subset])
  (folder/'summary.json').write_text(json.dumps(allout[cat],ensure_ascii=False,indent=2),encoding='utf-8')
  print(cat,json.dumps(allout[cat]['groups']), 'target',json.dumps(allout[cat]['target_subset']), 'threshold',cutoff,flush=True)
 metadata=dict(protocol='Same algorithm: connected components per original foreground label, 4x4 per-label pixel occupancy. Score = mean of relative directed count deficit and spatial occupancy deficit, minimized over whole normal training configurations; calibrate category-specific threshold by 99th percentile of normal leave-one-out scores. Existing per-category LogiCo segmentation labels required; no model training. Thresholds per class, algorithm fixed.',
 warning='Exploratory image-level feature-space score only. Whole-query segmentation is used to select closest reference; this is not masked-context prediction and cannot be treated as leakage-free E4 localization. Connected components may merge or fragment; colour labels make connector configurations mutually exclusive. Tests previously inspected; threshold training LOO has no independent test claim. No spatial evidence or AU-sPRO calculated.',categories=allout)
 (OUT/'summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
