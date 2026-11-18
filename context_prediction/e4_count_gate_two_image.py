"""Connector-specific two-image diagnostic, not a general detector."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from context_prediction.e4_full_line_mask_probe import ROOT,DATA,SEG,OUT as BANK,WEIGHT,digest,signature,visible_strips,rank,load_dinov3_model
from context_prediction.e4_normal_alignment_probe import read_components,anchors,connector_transform
OUT=ROOT/'saved_results/context_e4_count_gate_two_image'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 paths=sorted((DATA/'train/good').glob('*.png'))
 with np.load(BANK/'train_side_strips.npz') as z:
  names=z['names'];features=z['features'];labs=z['labs']
  if list(z['band'])!=[55,207] or str(z['weight_hash'])!=digest(WEIGHT) or list(z['hashes'])!=[digest(p) for p in paths] or list(names)!=[p.stem for p in paths]:raise ValueError('Reference cache mismatch')
 model=load_dinov3_model('dinov3_vitb16',pretrained_weight_path=str(WEIGHT)).cuda().eval()
 manifest={}
 for key,split,stem in [('missing_026','test/logical_anomalies','026'),('normal_000','validation/good','000')]:
  with Image.open(DATA/split/(stem+'.png')) as im:image=im.convert('RGB')
  f,lab=visible_strips(image,55,207,model);ids=rank(f,lab,features,labs,names)
  components=read_components(SEG/split/(stem+'.png'));connector=anchors(components)
  lines=[c for c in components.values() if c['class_id'] in (0,1,2)]
  masks=[];excluded=[]
  for refid in ids:
   ref,reason=signature(SEG/'train/good'/(refid+'.png'))
   if ref is None or connector is None:
    excluded.append(dict(reference=refid,reason=reason or 'query_connector_count'));continue
   result=connector_transform(ref['connectors'],connector)
   if result is None:excluded.append(dict(reference=refid,reason='invalid_transform'));continue
   mask=cv2.warpAffine(ref['line']['mask'].astype('uint8'),result[0],(256,256),flags=cv2.INTER_NEAREST)>0
   masks.append(mask)
  if not masks:raise ValueError('No valid reference masks for '+key)
  expected=np.mean(masks,axis=0); uncertain=connector is None
  evidence=expected if len(lines)==0 and not uncertain else np.zeros_like(expected)
  observed=np.any([v['mask'] for v in lines],axis=0) if lines else np.zeros((256,256),bool)
  np.savez_compressed(OUT/(key+'.npz'),expected=expected,observed=observed,evidence=evidence)
  panels=[('Original',image.resize((512,256))),('Reference occupancy 0-1',expected),('Observed wire segmentation',observed.astype(float)),('Count-gated evidence 0-1',evidence)]
  canvas=Image.new('RGB',(1024,572),'white');draw=ImageDraw.Draw(canvas)
  for i,(title,panel) in enumerate(panels):
   x=(i%2)*512;y=(i//2)*286;draw.text((x+5,y+7),title,fill='black')
   if isinstance(panel,np.ndarray):panel=Image.fromarray(cv2.cvtColor(cv2.applyColorMap(np.uint8(panel*255),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)).resize((512,256),Image.Resampling.NEAREST)
   canvas.paste(panel,(x,y+30))
  canvas.save(OUT/(key+'.png'))
  manifest[key]=dict(reference_ids=ids,usable_references=len(masks),excluded=excluded,wire_count=len(lines),connector_count=sum(c['class_id']==3 for c in components.values()),gate='uncertain' if uncertain else ('possible_missing' if len(lines)==0 else 'not_triggered'),evidence_sum=float(evidence.sum()),evidence_max=float(evidence.max()))
 manifest['limitations']='Exploratory connector-specific zero wire gate, not an E4 general method. Query retrieves outside fixed band 55:207, which may still expose actual wire outside the band. Registration uses full connector masks. Normal evidence is zero by construction. No checkpoint inference, ground truth localization metric, A fusion or independent evaluation.'
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
