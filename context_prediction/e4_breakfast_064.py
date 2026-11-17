"""Exploratory breakfast-box 064 occupancy check with original LogiCo segments."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from context_prediction.e3_evaluate import queries, tile_distances, best_ids
from context_prediction.tile_isolation import ROOT, WEIGHT, encode_tiles
from context_prediction.e4_two_image_precheck import component_set, in_box, match_same_class, missing_vote, visible_key
from dinov3.hub.backbones import load_dinov3_model

DATA=ROOT/'data/mvtec_loco/breakfast_box'
SEG=ROOT/'data/mvtec_loco_segmentations/breakfast_box'
OUT=ROOT/'saved_results/context_e4_breakfast_064'
NAMES=['peach','orange','cereals','banana chips','almonds','lunch box','black background']


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=sorted((DATA/'train/good').glob('*.png'))
    if len(train)!=351: raise RuntimeError('Unexpected training image count')
    model=load_dinov3_model('dinov3_vitb16',pretrained_weight_path=str(WEIGHT)).cuda().eval()
    cache=OUT/'train_tile_features.npz'; hashes=[sha(p) for p in train]; model_hash=sha(WEIGHT)
    if cache.exists():
        with np.load(cache) as f:
            if list(f['names'])!=[p.name for p in train] or list(f['hashes'])!=hashes or str(f['model_hash'])!=model_hash:
                raise ValueError('Reference cache inputs changed; rebuild cache')
            names=f['names']; feats=f['features']; labs=f['labs']
        print('Loaded 351 cached training images',flush=True)
    else:
        embeddings=[]; colors=[]
        for i,path in enumerate(train,1):
            with Image.open(path) as img: feat,lab=encode_tiles(img,model,'cuda:0')
            embeddings.append(feat); colors.append(lab)
            if i%25==0 or i==len(train): print('Training tile features',i,'/',len(train),flush=True)
        names=np.asarray([p.name for p in train]); feats=np.stack(embeddings); labs=np.stack(colors)
        tmp=cache.with_suffix('.tmp.npz'); np.savez_compressed(tmp,names=names,hashes=hashes,model_hash=model_hash,features=feats,labs=labs); tmp.replace(cache)
    img_path=DATA/'test/logical_anomalies/064.png'; seg_path=SEG/'test/logical_anomalies/064.png'
    with Image.open(img_path) as im:image=im.convert('RGB')
    retrieval=[]
    for name,hidden,box in queries():
        visible=[i for i in range(16) if i not in hidden]
        query=visible_key(image,visible,model)
        retrieval.append((name,box,best_ids(tile_distances(query,feats,labs),names,visible)))
    observed,seg=component_set(img_path,seg_path,model,class_ids=range(6))
    observed_map=np.zeros((256,256),bool)
    for item in observed: observed_map|=item['mask']
    expected_map=np.zeros((256,256),np.float32); evidence=np.zeros((256,256),np.float32)
    cached={}; records=[]
    for window,box,refs in retrieval:
        x1,y1,x2,y2=[round(x*256) for x in box]; clip=np.zeros((256,256),bool);clip[y1:y2,x1:x2]=True
        actual=in_box(observed,box); expected_votes=[];missing_votes=[]; details=[]
        for ref in refs:
            stem=Path(ref).stem
            if stem not in cached: cached[stem]=component_set(DATA/'train/good'/ref,SEG/'train/good'/ref,model,class_ids=range(6))[0]
            expected=in_box(cached[stem],box); pairs,missing=match_same_class(expected,actual)
            expected_vote,missing_vote_map=missing_vote(expected,actual,pairs,clip)
            expected_votes.append(expected_vote);missing_votes.append(missing_vote_map)
            details.append({'reference':stem,'expected':len(expected),'observed':len(actual),'matched':len(pairs),'unmatched_expected':len(missing)})
        expected_map=np.maximum(expected_map,np.mean(expected_votes,axis=0));evidence=np.maximum(evidence,np.mean(missing_votes,axis=0))
        records.append({'window':window,'box_xyxy':box,'reference_ids':[Path(r).stem for r in refs],'matches':details})
    np.savez_compressed(OUT/'breakfast_064.npz',expected=expected_map,observed=observed_map,evidence=evidence,original_seg=seg)
    report=dict(image=str(img_path),image_sha256=sha(img_path),segmentation=str(seg_path),segmentation_sha256=sha(seg_path),
                dataset_annotation='0_nectarines_3_tangerines',reference_count=len(train),dino_sha256=model_hash,
                description='exploratory, 20 E3 windows, 5 per-window references; original segmentation, same-class matching; no geometric registration',
                observed_counts={NAMES[i]:sum(x['class_id']==i for x in observed) for i in range(6)},windows=records)
    (OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    source=image.resize((512,410),Image.Resampling.BILINEAR); panels=[source]
    for arr in (expected_map,observed_map,evidence):
        m=cv2.applyColorMap(np.uint8(np.clip(arr,0,1)*255),cv2.COLORMAP_INFERNO)
        m=cv2.cvtColor(cv2.resize(m,(512,410),interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2RGB)
        panels.append(Image.fromarray(m))
    canvas=Image.new('RGB',(1024,868),'white'); draw=ImageDraw.Draw(canvas)
    for i,(panel,label) in enumerate(zip(panels,['Original','Expected occupancy (0-1)','Observed occupancy','Missing evidence (0-1)'])):
        x=(i%2)*512;y=(i//2)*434;canvas.paste(panel,(x,y+24));draw.text((x+5,y+3),label,fill='black')
    canvas.save(OUT/'breakfast_064.png')
    print(json.dumps({'observed_counts':report['observed_counts'],'evidence_max':float(evidence.max()),'output':str(OUT)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
