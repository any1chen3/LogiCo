"""Normal-only instance split gate calibrated from same-label normal components."""
import csv,json
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,yaml
from PIL import Image,ImageDraw
from context_prediction.e4_instance_shape_gate import gated
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'saved_results/context_e4_instance_reference_area'

def labels_at(p):
 with Image.open(p) as im:return np.asarray(im.resize((256,256),Image.Resampling.NEAREST))

def components(mask):
 n,cc,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
 return [(cc==i,int(stats[i,4])) for i in range(1,n) if stats[i,4]>=4]

def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[];models={}
 for cat in ('breakfast_box','splicing_connectors'):
  names=yaml.safe_load((ROOT/'configs/segmentations/mvtec_loco'/f'{cat}.yaml').read_text(encoding='utf-8'))['prompts'][:-1]
  seg=ROOT/'data/mvtec_loco_segmentations'/cat
  train=sorted((seg/'train/good').glob('*.png'));area=defaultdict(list);source=defaultdict(int)
  for p in train:
   z=labels_at(p)
   for i in range(len(names)):
    cs=components(z==i)
    if len(cs)>=2:
     source[i]+=1;area[i].extend(a for _,a in cs)
  models[cat]={names[i]:dict(training_multi_region_images=source[i],median_area=float(np.median(area[i])) if area[i] else None,calibration_samples=len(area[i])) for i in range(len(names))}
  for split in ('train/good','validation/good'):
   for p in sorted((seg/split).glob('*.png')):
    z=labels_at(p)
    for cls,name in enumerate(names):
     oldparts=components(z==cls); med=models[cat][name]['median_area'];newcount=0;out=np.zeros(z.shape,dtype=np.int32);accepted=0;rejected=0
     for comp,npx in oldparts:
      _,n,_,pieces=gated(comp);candidate=[int((pieces==j).sum()) for j in range(1,n+1)]
      # Same rule for all labels; component reference comes only from normal train.
      valid=(n>=2 and med is not None and source[cls]>=10 and npx>=1.5*med and all(.5*med<=a<=1.8*med for a in candidate))
      if valid:
       accepted+=1
       for j in range(1,n+1):newcount+=1;out[pieces==j]=newcount
      else:
       if n>=2:rejected+=1
       newcount+=1;out[comp]=newcount
     rows.append(dict(category=cat,split=split,image=p.name,label=name,before=len(oldparts),after=newcount,accepted=accepted,rejected=rejected,training_median_area=med))
     if (cat,split,p.stem,name) in [('breakfast_box','train/good','239','orange'),('breakfast_box','train/good','337','orange'),('splicing_connectors','validation/good','003','connector')]:
      rgb=np.zeros((256,256,3),dtype='uint8')
      for j in range(1,newcount+1):rgb[out==j]=[(83*j)%255,(149*j)%255,(211*j)%255]
      with Image.open(ROOT/'data/mvtec_loco'/cat/split/p.name) as im:orig=im.convert('RGB').resize((512,256))
      img=Image.new('RGB',(1024,286),'white');img.paste(orig,(0,30));img.paste(Image.fromarray(rgb).resize((512,256),Image.Resampling.NEAREST),(512,30));d=ImageDraw.Draw(img);d.text((5,7),cat+' '+p.stem,fill='black');d.text((517,7),f'{name}: {len(oldparts)} to {newcount}',fill='black');img.save(OUT/(cat+'_'+p.stem+'.png'))
 with (OUT/'per_image_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 result={}
 for cat in models:
  result[cat]={}
  for split in ('train/good','validation/good'):
   result[cat][split]={}
   for name in models[cat]:
    local=[r for r in rows if r['category']==cat and r['split']==split and r['label']==name]
    result[cat][split][name]=dict(images=len(local),split=sum(r['after']>r['before'] for r in local),rejected=sum(r['rejected']>0 for r in local))
   print(cat,split,[(k,v['split']) for k,v in result[cat][split].items()],flush=True)
  print(cat,'models',models[cat],flush=True)
 summary=dict(protocol='Normal-training same-label area prior; use original masks with at least 2 components to estimate median per label, require at least 10 such normal images. Apply prior shape-gated watershed only if parent area >=1.5 median and every proposed piece lies in [0.5,1.8] median. Same numeric rule for all categories.',warning='Exploratory heuristic; fragmented connected components are not verified object instances, median may be contaminated. Training is evaluated in-sample; validation has been previously inspected, no independent physical instance GT. Not anomaly/localization performance. Thresholds chosen after previous results and no test-sample inference.',models=models,results=result)
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
