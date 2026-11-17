"""E4 exploratory precheck with cached E3 retrieval and original LogiCo segments."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from context_prediction.e3_evaluate import CACHE, queries, read, tile_distances, best_ids
from context_prediction.tile_isolation import ROOT, WEIGHT, MEAN, STD, boxes
from context_prediction.evidence import component_cost
from dinov3.hub.backbones import load_dinov3_model

CATEGORY='splicing_connectors'
DATA=ROOT/'data/mvtec_loco'/CATEGORY
SEG=ROOT/'data/mvtec_loco_segmentations'/CATEGORY
OUT=ROOT/'saved_results/context_e4_two_image_precheck'
CHECKPOINT=ROOT/'saved_results/mvtec_loco'/CATEGORY/'LogiCo_best_ckpt.pkl'
NAMES=['blue cable','yellow cable','red cable','connector','grid']


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def component_set(image_path, segmentation_path, model):
    """Connected components in original class map; background class omitted."""
    with Image.open(image_path) as im: rgb=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    with Image.open(segmentation_path) as im: seg=np.asarray(im.resize((256,256),Image.Resampling.NEAREST))
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32); lab[:,:,0]*=100/255; lab[:,:,1:]-=128
    items=[]; crops=[]
    for cls in range(4):
        n, labels, stats, centroids=cv2.connectedComponentsWithStats(np.uint8(seg==cls),8)
        for index in range(1,n):
            x,y,w,h,area=stats[index]
            if area < 4: continue  # 16 pixels on 512x512 evaluation grid
            mask=labels==index; cx,cy=centroids[index]
            crop=Image.fromarray(rgb[y:y+h,x:x+w]).resize((224,224),Image.Resampling.BICUBIC)
            crops.append(TF.normalize(TF.to_tensor(crop),MEAN,STD))
            items.append(dict(id=f'{cls}:{index}',class_id=cls,mask=mask,center=[(cx+.5)/256,(cy+.5)/256],
                              area=float(area)/(256*256),color=np.median(lab[mask],axis=0)))
    with torch.inference_mode():
        for start in range(0,len(crops),4):
            vec=F.normalize(model.forward_features(torch.stack(crops[start:start+4]).cuda())['x_norm_clstoken'],dim=1).cpu().numpy()
            for item,feature in zip(items[start:start+4],vec): item['appearance']=feature
    return items,seg


def match_same_class(expected,observed):
    """Same original label is required; virtual penalties use E1's 0.35."""
    m,n=len(expected),len(observed)
    if not m: return {},[]
    if not n: return {},list(range(m))
    costs=np.full((m+n,m+n),np.inf)
    for i,e in enumerate(expected):
        for j,o in enumerate(observed):
            if e['class_id']==o['class_id']: costs[i,j]=component_cost(e,o)
        costs[i,n+i]=.35
    for j in range(n): costs[m+j,j]=.35
    costs[m:,n:]=0
    rr,cc=linear_sum_assignment(costs)
    return {int(i):int(j) for i,j in zip(rr,cc) if i<m and j<n},[int(i) for i,j in zip(rr,cc) if i<m and j>=n]


def in_box(items,box):
    x1,y1,x2,y2=box
    return [x for x in items if x1<=x['center'][0]<x2 and y1<=x['center'][1]<y2]


def missing_vote(expected, observed, pairs, clip):
    missing_map=np.zeros(clip.shape,bool); expected_map=np.zeros(clip.shape,bool)
    for i,e in enumerate(expected):
        mask=e['mask'] & clip; expected_map |= mask
        if i in pairs:
            actual=observed[pairs[i]]['mask'].astype(np.uint8)
            expanded=cv2.dilate(actual,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
            missing_map |= mask & ~expanded
        else: missing_map |= mask
    return expected_map,missing_map


def visible_key(image, visible, model):
    """Crop and encode only tiles outside the current target window."""
    grid=boxes(image.width,image.height); crops=[]; colors=[]
    for idx in visible:
        rgb=np.asarray(image.crop(grid[idx]).convert('RGB'))
        lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32); lab[...,0]*=100/255; lab[...,1:]-=128
        colors.append(np.median(lab.reshape(-1,3),axis=0))
        crop=Image.fromarray(rgb).resize((224,224),Image.Resampling.BICUBIC)
        crops.append(TF.normalize(TF.to_tensor(crop),MEAN,STD))
    features=[]
    with torch.inference_mode():
        for start in range(0,len(crops),4):
            features.append(F.normalize(model.forward_features(torch.stack(crops[start:start+4]).cuda())['x_norm_clstoken'],dim=1).cpu().numpy())
    result_features=np.zeros((16,768),dtype=np.float32); result_lab=np.zeros((16,3),dtype=np.float32)
    result_features[visible]=np.concatenate(features); result_lab[visible]=np.asarray(colors)
    return {'tile_features':result_features,'tile_lab':result_lab}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bankpaths=sorted((CACHE/CATEGORY/'train').glob('*.npz'))
    if len(bankpaths)!=360: raise RuntimeError('The E3 training reference cache is incomplete')
    bank=[(p,read(p)) for p in bankpaths]
    names=np.array([p.name for p,_ in bank]); feats=np.stack([d['tile_features'] for _,d in bank]); labs=np.stack([d['tile_lab'] for _,d in bank])
    model=load_dinov3_model('dinov3_vitb16',pretrained_weight_path=str(WEIGHT)).cuda().eval()
    samples=[('test_missing_026',DATA/'test/logical_anomalies/026.png',SEG/'test/logical_anomalies/026.png'),
             ('validation_good_000',DATA/'validation/good/000.png',SEG/'validation/good/000.png')]
    component_cache={}; manifest={'checkpoint':str(CHECKPOINT),'checkpoint_sha256':digest(CHECKPOINT),
      'matching':'0.5 appearance + 0.2 Lab + 0.15 center + 0.15 area, same segmentation class only, unmatched=0.35',
      'windows':'E3 16 small and 4 large; component centroid assignment, 256x256 clip; max over windows',
      'vote':'5 per-reference missing masks average; matched missing=expected minus 2px dilation of matched observed',
      'normal_selection':'validation/good/000 visually has similar multi-lever connectors; missing cable color not known',
      'samples':{}}
    for sample_id,image_path,seg_path in samples:
        # Retrieval uses only independently encoded RGB tiles, before segmentations are read.
        with Image.open(image_path) as im: image=im.convert('RGB')
        retrieval=[]
        for name,hidden,box in queries():
            visible=[i for i in range(16) if i not in hidden]
            query=visible_key(image,visible,model)
            distances=tile_distances(query,feats,labs)
            retrieval.append((name,box,best_ids(distances,names,visible)))
        observed,seg=component_set(image_path,seg_path,model)
        out_expected=np.zeros((256,256),np.float32); out_observed=np.zeros((256,256),np.float32); evidence=np.zeros((256,256),np.float32)
        logs=[]
        for name,box,refs in retrieval:
            x1,y1,x2,y2=[int(round(z*256)) for z in box]
            clip=np.zeros((256,256),bool); clip[y1:y2,x1:x2]=True
            visible_observed=in_box(observed,box)
            obs_union=np.zeros((256,256),bool)
            for item in visible_observed: obs_union |= item['mask'] & clip
            out_observed=np.maximum(out_observed,obs_union)
            votes=[]; occupancy=[]; detail=[]
            for ref in refs:
                stem=Path(ref).stem
                if stem not in component_cache:
                    component_cache[stem]=component_set(DATA/'train/good'/(stem+'.png'),SEG/'train/good'/(stem+'.png'),model)[0]
                expected=in_box(component_cache[stem],box)
                pairs,missing=match_same_class(expected,visible_observed)
                expected_map,missing_map=missing_vote(expected,visible_observed,pairs,clip)
                occupancy.append(expected_map); votes.append(missing_map)
                detail.append({'reference':stem,'expected':len(expected),'observed':len(visible_observed),'matched':len(pairs),'unmatched_expected':len(missing)})
            out_expected=np.maximum(out_expected,np.mean(occupancy,axis=0))
            evidence=np.maximum(evidence,np.mean(votes,axis=0))
            logs.append({'window':name,'box_xyxy':box,'reference_ids':[Path(x).stem for x in refs],'matches':detail})
        np.savez_compressed(OUT/(sample_id+'.npz'),expected=out_expected,observed=out_observed,evidence=evidence,original_seg=seg)
        manifest['samples'][sample_id]={'image':str(image_path),'segmentation':str(seg_path),'observed_counts_by_label':{NAMES[i]:sum(x['class_id']==i for x in observed) for i in range(4)},'windows':logs,'evidence_max':float(evidence.max())}
        with Image.open(image_path) as im: orig=im.convert('RGB').resize((512,256),Image.Resampling.BILINEAR)
        panels=[orig]
        for arr in (out_expected,out_observed,evidence):
            heat=cv2.applyColorMap(np.uint8(np.clip(arr,0,1)*255),cv2.COLORMAP_INFERNO)
            heat=cv2.cvtColor(cv2.resize(heat,(512,256),interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2RGB)
            panels.append(Image.fromarray(heat))
        canvas=Image.new('RGB',(1024,568),'white'); draw=ImageDraw.Draw(canvas)
        for i,(panel,label) in enumerate(zip(panels,['Original','Expected occupancy (0-1)','Observed occupancy','Missing evidence (0-1)'])):
            x=(i%2)*512; y=(i//2)*284; canvas.paste(panel,(x,y+24)); draw.text((x+5,y+3),label,fill='black')
        canvas.save(OUT/(sample_id+'.png'))
        print(sample_id,'regions',len(observed),'reference sets',len(logs),'evidence max',float(evidence.max()),flush=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
