"""Normal-only exploratory watershed split for all foreground segmentation classes."""
import csv,json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'saved_results/context_e4_instance_split_normal'

def split_mask(mask):
 n,cc,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
 output=np.zeros(mask.shape,dtype=np.int32);nold=0;newid=0
 for index in range(1,n):
  if stats[index,4]<4:continue
  nold+=1; comp=cc==index; dist=ndi.distance_transform_edt(comp)
  points=peak_local_max(dist,labels=comp.astype('uint8'),min_distance=12,threshold_abs=5,exclude_border=False)
  if len(points)<2:
   newid+=1;output[comp]=newid;continue
  markers=np.zeros(comp.shape,dtype=np.int32)
  for j,(y,x) in enumerate(points,1):markers[y,x]=j
  regions=watershed(-dist,markers,mask=comp)
  for j in range(1,len(points)+1):
   newid+=1;output[regions==j]=newid
 return nold,newid,output

def main():
 OUT.mkdir(parents=True,exist_ok=True);records=[]
 for cat in ('splicing_connectors','breakfast_box'):
  seg=ROOT/'data/mvtec_loco_segmentations'/cat; cfg=yaml.safe_load((ROOT/'configs/segmentations/mvtec_loco'/f'{cat}.yaml').read_text(encoding='utf-8'));names=cfg['prompts']
  if names[-1].lower() not in ('grid','black background'):raise ValueError('unexpected background')
  for split in ('train/good','validation/good'):
   for p in sorted((seg/split).glob('*.png')):
    with Image.open(p) as im:labels=np.asarray(im.resize((256,256),Image.Resampling.NEAREST))
    for cls,name in enumerate(names[:-1]):
     old,new,instances=split_mask(labels==cls)
     records.append(dict(category=cat,split=split,image=p.name,label=name,old_count=old,new_count=new,delta=new-old))
     if cat=='breakfast_box' and split=='train/good' and p.stem in ('239','337') and name=='orange':
      rgb=np.zeros((*instances.shape,3),dtype='uint8')
      for k in range(1,new+1):rgb[instances==k]=[(37*k)%255,(149*k)%255,(230*k)%255]
      with Image.open(ROOT/'data/mvtec_loco'/cat/split/p.name) as im:original=im.convert('RGB').resize((512,512))
      canvas=Image.new('RGB',(1024,542),'white');canvas.paste(original,(0,30));canvas.paste(Image.fromarray(rgb).resize((512,512),Image.Resampling.NEAREST),(512,30))
      draw=ImageDraw.Draw(canvas);draw.text((5,7),'Normal '+p.stem,fill='black');draw.text((517,7),f'Orange: before {old}, after {new}',fill='black');canvas.save(OUT/(p.stem+'_orange.png'))
 with (OUT/'per_image_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
 groups={}
 for cat in ('splicing_connectors','breakfast_box'):
  groups[cat]={}
  for split in ('train/good','validation/good'):
   subset=[r for r in records if r['category']==cat and r['split']==split]
   groups[cat][split]={}
   for label in sorted(set(r['label'] for r in subset)):
    group=[r for r in subset if r['label']==label];present=[r for r in group if r['old_count']>0]
    groups[cat][split][label]=dict(images=len(group),present=len(present),images_split=sum(r['delta']>0 for r in group),old_distribution=dict(sorted(Counter(str(r['old_count']) for r in group).items())),new_distribution=dict(sorted(Counter(str(r['new_count']) for r in group).items())))
 summary=dict(protocol='Same distance-transform watershed on every original foreground label; 256x256, components >=4 pixels, local maxima min_distance=12, threshold_abs=5, otherwise keep component. No per-category tuning. Normal train and validation only; examples 239 and 337 were identified by prior test-sample analysis.',
  warning='Exploratory non-semantic geometric oversegmentation: increased count is not automatically correct, no independent instance GT or recall/precision assessed. These parameters are illustrative, not frozen or tuned. Small thin wires may never be eligible for splitting.',groups=groups)
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 for cat in groups:
  for split in groups[cat]:
   print(cat,split,[(k,v['images_split'],v['present']) for k,v in groups[cat][split].items()],flush=True)
if __name__=='__main__':main()
