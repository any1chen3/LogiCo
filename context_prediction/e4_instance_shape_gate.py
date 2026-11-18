"""Normal-only class-agnostic shape-gated watershed exploratory probe."""
import csv,json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import yaml
from context_prediction.e4_instance_split_normal import split_mask
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'saved_results/context_e4_instance_shape_gate'
# Fixed exploratory thresholds on 256x256 masks; apply to all labels, including containers.
CIRCULARITY_MIN=.20;SOLIDITY_MIN=.75;ASPECT_MAX=2.5

def gated(mask):
 n,cc,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
 out=np.zeros(mask.shape,np.int32);old=0;total=0;eligible=0
 for i in range(1,n):
  x,y,w,h,area=stats[i]
  if area<4:continue
  old+=1;comp=cc==i
  contours,_=cv2.findContours(comp.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  if not contours:continue
  contour=max(contours,key=cv2.contourArea);perimeter=cv2.arcLength(contour,True)
  hull=cv2.convexHull(contour);hull_area=cv2.contourArea(hull)
  contour_area=cv2.contourArea(contour)
  circ=4*np.pi*contour_area/max(perimeter*perimeter,1e-9);sol=contour_area/max(hull_area,1e-9)
  aspect=max(w,h)/max(min(w,h),1)
  can=bool(circ>=CIRCULARITY_MIN and sol>=SOLIDITY_MIN and aspect<=ASPECT_MAX)
  if can:
   eligible+=1;_,num,instances=split_mask(comp)
  else:num=1;instances=comp.astype('int32')
  for j in range(1,num+1):total+=1;out[instances==j]=total
 return old,total,eligible,out

def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[]
 for cat in ('splicing_connectors','breakfast_box'):
  labels=yaml.safe_load((ROOT/'configs/segmentations/mvtec_loco'/f'{cat}.yaml').read_text(encoding='utf-8'))['prompts']
  if labels[-1].lower() not in ('grid','black background'):raise ValueError('Unexpected background')
  for split in ('train/good','validation/good'):
   for p in sorted((ROOT/'data/mvtec_loco_segmentations'/cat/split).glob('*.png')):
    with Image.open(p) as im:seg=np.asarray(im.resize((256,256),Image.Resampling.NEAREST))
    for i,name in enumerate(labels[:-1]):
     old,new,eligible,inst=gated(seg==i);rows.append(dict(category=cat,split=split,image=p.name,label=name,before=old,after=new,eligible_components=eligible,added=new-old))
     if (cat,split,p.stem,name) in [('breakfast_box','train/good','239','orange'),('breakfast_box','train/good','337','orange'),('splicing_connectors','train/good','000','yellow cable')]:
      colors=np.zeros((256,256,3),dtype='uint8')
      for j in range(1,new+1):colors[inst==j]=[(83*j)%255,(149*j)%255,(211*j)%255]
      with Image.open(ROOT/'data/mvtec_loco'/cat/split/p.name) as im:orig=im.convert('RGB').resize((512,256))
      canvas=Image.new('RGB',(1024,286),'white');canvas.paste(orig,(0,30));canvas.paste(Image.fromarray(colors).resize((512,256),Image.Resampling.NEAREST),(512,30));draw=ImageDraw.Draw(canvas);draw.text((5,7),cat+' '+p.stem,fill='black');draw.text((517,7),f'{name}: {old} to {new}',fill='black');canvas.save(OUT/(cat+'_'+p.stem+'.png'))
 with (OUT/'per_image_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 summary={}
 for cat in ('splicing_connectors','breakfast_box'):
  summary[cat]={}
  for split in ('train/good','validation/good'):
   subset=[r for r in rows if r['category']==cat and r['split']==split];result={}
   for name in sorted({r['label'] for r in subset}):
    group=[r for r in subset if r['label']==name];result[name]=dict(images=len(group),present=sum(r['before']>0 for r in group),images_split=sum(r['added']>0 for r in group),before=dict(Counter(str(r['before']) for r in group)),after=dict(Counter(str(r['after']) for r in group)))
   summary[cat][split]=result
   print(cat,split,[(n,v['images_split'],v['present']) for n,v in result.items()],flush=True)
 (OUT/'summary.json').write_text(json.dumps(dict(protocol='Same geometry gate for all labels: circularity >=0.20, convex-hull solidity >=0.75, aspect ratio <=2.5 before prior watershed; 256x256. No test data or model training.',warning='Thresholds chosen heuristically after prior test-related diagnosis; normal validation was previously reviewed. Exploratory, no independent physical instance labels; splitting frequency is not correctness.',results=summary),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
