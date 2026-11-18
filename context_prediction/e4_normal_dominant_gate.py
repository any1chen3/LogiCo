"""Explore normal-trained class-independent dominant-region exclusion."""
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'saved_results/context_e4_instance_shape_gate/per_image_class.csv'
OUT=ROOT/'saved_results/context_e4_normal_dominant_gate'
MIN_MEDIAN_AREA=.25
MIN_PRESENCE=.90

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with SRC.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 metadata={}
 for cat in ('splicing_connectors','breakfast_box'):
  names=yaml.safe_load((ROOT/'configs/segmentations/mvtec_loco'/f'{cat}.yaml').read_text(encoding='utf-8'))['prompts'][:-1]
  with np.load(ROOT/'saved_results/context_e4_shared_normal_configuration'/cat/'normal_bank.npz') as bank:
   areas=bank['spatial_occupancy'].reshape(-1,len(names),16).mean(axis=2)
   counts=bank['counts']
  metadata[cat]={}
  for i,name in enumerate(names):
   median=float(np.median(areas[:,i]));frequency=float(np.mean(counts[:,i]>0))
   metadata[cat][name]=dict(training_median_fraction=median,training_presence=frequency,dominant=bool(median>=MIN_MEDIAN_AREA and frequency>=MIN_PRESENCE))
  for r in rows:
   if r['category']!=cat:continue
   dominant=metadata[cat][r['label']]['dominant'];r['dominant_excluded']=dominant
   r['after_dominant_gate']=int(r['before']) if dominant else int(r['after'])
   r['split_after_dominant_gate']=r['after_dominant_gate']>int(r['before'])
 with (OUT/'per_image_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 result={}
 for cat in metadata:
  result[cat]={}
  for split in ('train/good','validation/good'):
   group=[r for r in rows if r['category']==cat and r['split']==split];result[cat][split]={}
   for name in metadata[cat]:
    vals=[r for r in group if r['label']==name]
    result[cat][split][name]=dict(images=len(vals),present=sum(int(r['before'])>0 for r in vals),split_before=sum(int(r['added'])>0 for r in vals),split_after=sum(r['split_after_dominant_gate'] for r in vals))
   print(cat,split,[(k,v['split_after']) for k,v in result[cat][split].items()])
 summary=dict(protocol='Same class-independent normal-trained dominant-region heuristic: exclude labels whose median area across normal training images >=0.25 of image and presence >=0.90. Apply gate to prior shape-gated watershed for normal train and validation; no anomaly images used in this experiment. Original category segmentation labels still required.',
 warning='Heuristic chosen after inspecting prior results and normal bank, not independently validated or a semantic container detector; may suppress large or persistent actual objects and miss small containers. Same validation sets previously inspected. No instance ground truth, localization or anomaly metrics.', thresholds=dict(median_area=MIN_MEDIAN_AREA,presence=MIN_PRESENCE),training_label_statistics=metadata,results=result)
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print('excluded', {cat:[name for name,v in vals.items() if v['dominant']] for cat,vals in metadata.items()})
if __name__=='__main__':main()
