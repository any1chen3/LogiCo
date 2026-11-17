"""Explore matched line displacement and residual on ten normal validation images."""
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from context_prediction.e3_evaluate import CACHE, ROOT, queries, read, tile_distances, best_ids
from context_prediction.e4_two_image_precheck import WEIGHT, component_set, match_same_class, missing_vote
from context_prediction.e4_window_intersection import intersecting
from context_prediction.evidence import component_cost
from dinov3.hub.backbones import load_dinov3_model

CAT='splicing_connectors'
DATA=ROOT/'data/mvtec_loco'/CAT
SEG=ROOT/'data/mvtec_loco_segmentations'/CAT
OUT=ROOT/'saved_results/context_e4_normal_line_audit'


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def overlay(image,ref,obs,clip):
    """Red=reference, green=observed, yellow=intersection; restricted to window."""
    ref=ref & clip; obs=obs & clip
    rgb=np.zeros((256,256,3),np.uint8);rgb[:,:,0]=ref*255;rgb[:,:,1]=obs*255
    colored=Image.fromarray(rgb).resize(image.size,Image.Resampling.NEAREST)
    return Image.blend(image,colored,.55)


def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'examples').mkdir(exist_ok=True)
    paths=sorted((CACHE/CAT/'train').glob('*.npz'))
    if len(paths)!=360:raise RuntimeError('Training reference cache incomplete')
    bank=[(p,read(p)) for p in paths]
    names=np.asarray([p.stem for p,_ in bank]); features=np.stack([d['tile_features'] for _,d in bank]); labs=np.stack([d['tile_lab'] for _,d in bank])
    conf=json.loads((CACHE/'config.json').read_text(encoding='utf-8'))
    if conf['dino_sha256']!=sha(WEIGHT):raise RuntimeError('DINO differs from E3 reference cache')
    model=load_dinov3_model('dinov3_vitb16',pretrained_weight_path=str(WEIGHT)).cuda().eval()
    cache={}; allpairs=[]; allwindows=[]; images=[]; detail={}
    for img_path in sorted((DATA/'validation/good').glob('*.png'))[:10]:
        stem=img_path.stem; query=read(CACHE/CAT/'validation'/(stem+'.npz'))
        if str(query['source_sha256'])!=sha(img_path):raise RuntimeError('Validation cache image changed')
        selections=[]
        for wname,hidden,box in queries():
            visible=[i for i in range(16) if i not in hidden]
            selections.append((wname,box,best_ids(tile_distances(query,features,labs),names,visible)))
        actual,seg=component_set(img_path,SEG/'validation/good'/img_path.name,model)
        line_actual=[x for x in actual if x['class_id'] in (0,1,2)]
        line_mask=np.isin(seg,[0,1,2]); fused=np.zeros((256,256),np.float32)
        candidates=[]; image_pairs=[]; window_records=[]
        for wname,box,refs in selections:
            x1,y1,x2,y2=[round(v*256) for v in box]; clip=np.zeros((256,256),bool);clip[y1:y2,x1:x2]=True
            observed=intersecting(actual,clip); votes=[]
            for ref_id in refs:
                if ref_id not in cache:
                    cache[ref_id]=component_set(DATA/'train/good'/(ref_id+'.png'),SEG/'train/good'/(ref_id+'.png'),model)[0]
                expected=intersecting(cache[ref_id],clip); pairs,unmatched=match_same_class(expected,observed)
                _,full_missing=missing_vote(expected,observed,pairs,np.ones((256,256),bool));votes.append(full_missing & clip)
                lines=[(i,e) for i,e in enumerate(expected) if e['class_id'] in (0,1,2)]
                obs_lines=[x for x in observed if x['class_id'] in (0,1,2)]
                for idx,e in lines:
                    eclip=e['mask']&clip; epixels=int(eclip.sum()); hit=idx in pairs
                    row=dict(image=stem,window=wname,reference=ref_id,expected_id=e['id'],expected_color=e['class_id'],
                             observed_lines=len(obs_lines),observed_same_color=sum(o['class_id']==e['class_id'] for o in obs_lines),
                             status='matched' if hit else 'unmatched',expected_pixels_in_window=epixels)
                    if hit:
                        o=observed[pairs[idx]]; obsmask=o['mask']; expanded=cv2.dilate(obsmask.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
                        remain=eclip & ~expanded; inter=int(np.logical_and(e['mask'],obsmask).sum());union=int(np.logical_or(e['mask'],obsmask).sum())
                        dx=(e['center'][0]-o['center'][0])*256;dy=(e['center'][1]-o['center'][1])*256
                        row.update(observed_id=o['id'],cost=float(component_cost(e,o)),iou_full=inter/union,
                                   dx_256=float(dx),dy_256=float(dy),remaining_pixels=int(remain.sum()),remaining_fraction=float(remain.sum()/max(1,epixels)))
                        if remain.any(): candidates.append((int(remain.sum()),wname,ref_id,e['mask'].copy(),obsmask.copy(),clip.copy(),remain.copy()))
                    else:
                        row.update(observed_id='',cost='',iou_full='',dx_256='',dy_256='',remaining_pixels=epixels,remaining_fraction=1.)
                    allpairs.append(row);image_pairs.append(row)
                window_records.append(dict(window=wname,reference=ref_id,expected_line_count=len(lines),observed_line_count=len(obs_lines),
                                           expected_line_matched=sum(i in pairs for i,_ in lines)))
            fused=np.maximum(fused,np.mean(votes,axis=0)); allwindows.append(dict(image=stem,window=wname,references=';'.join(refs),observed_line_count=len([o for o in observed if o['class_id'] in (0,1,2)])))
        paired=[r for r in image_pairs if r['status']=='matched']
        item=dict(image=stem,line_components=len(line_actual),line_pixels=int(line_mask.sum()),
                  evidence_mean_on_line=float(fused[line_mask].mean()) if line_mask.any() else None,
                  evidence_nonzero_on_line=float((fused[line_mask]>0).mean()) if line_mask.any() else None,
                  expected_line_instances=len(image_pairs),matched=len(paired),unmatched=len(image_pairs)-len(paired),
                  median_pair_iou=float(np.median([r['iou_full'] for r in paired])) if paired else None,
                  median_remaining_fraction=float(np.median([r['remaining_fraction'] for r in paired])) if paired else None)
        images.append(item);detail[stem]={'image_sha256':sha(img_path),'observed_line_components':[{'id':c['id'],'class':c['class_id'],'center':c['center']} for c in line_actual],'reference_windows':window_records}
        np.savez_compressed(OUT/'examples'/(stem+'.npz'),evidence=fused,original_seg=seg)
        if candidates:
            # Exploratory illustration: matched reference pair with largest residual area.
            _,win,ref,expected_mask,obs_mask,clip,remain=sorted(candidates,key=lambda c:(-c[0],c[1],c[2]))[0]
            with Image.open(img_path) as im: orig=im.convert('RGB').resize((512,256),Image.Resampling.BILINEAR)
            panel=overlay(orig,expected_mask,obs_mask,clip)
            def mask_image(x):return Image.fromarray(np.uint8(x)*255).resize((512,256),Image.Resampling.NEAREST).convert('RGB')
            canv=Image.new('RGB',(1536,284),'white');draw=ImageDraw.Draw(canv)
            for j,(im,label) in enumerate([(orig,'Normal image '+stem),(panel,'Reference red / actual green; '+win+' ref '+ref),(mask_image(remain),'Uncovered reference after 2px tolerance')]):
                canv.paste(im,(j*512,28));draw.text((j*512+5,4),label,fill='black')
            canv.save(OUT/'examples'/(stem+'.png'))
        print('normal',stem,'line_components',len(line_actual),'matched',len(paired),'unmatched',len(image_pairs)-len(paired),flush=True)
    for filename,rows in [('line_pairs.csv',allpairs),('reference_windows.csv',allwindows),('image_summary.csv',images)]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'details.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps({'protocol':'first 10 sorted normal validation images; reference retrieval from train; repeated window-reference observations are not independent; no alignment or parameter change', 'images':images},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
