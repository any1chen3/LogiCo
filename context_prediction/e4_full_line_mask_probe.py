"""Normal-only exploratory fixed-band, side-strip reference retrieval probe."""
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from context_prediction.e3_evaluate import CACHE,ROOT,read,tile_distances,best_ids
from context_prediction.e4_normal_alignment_probe import read_components,anchors,end_height
from context_prediction.tile_isolation import WEIGHT,MEAN,STD
from dinov3.hub.backbones import load_dinov3_model

CAT='splicing_connectors'; DATA=ROOT/'data/mvtec_loco'/CAT
SEG=ROOT/'data/mvtec_loco_segmentations'/CAT
OUT=ROOT/'saved_results/context_e4_full_line_mask_probe'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def signature(path):
    components=read_components(path);con=anchors(components);lines=[c for c in components.values() if c['class_id'] in (0,1,2)]
    if con is None:return None,'not_two_connectors'
    if len(lines)!=1:return None,'not_one_line'
    end=end_height(lines[0]['mask'],con)
    if end is None:return None,'invalid_endpoints'
    return dict(line=lines[0],connectors=con,heights=end),None


def visible_strips(image,left,right,model):
    """Encode fixed strips only; target central band is never cropped."""
    w,h=image.size; xs=[(0,w*left//256),(w*right//256,w)]; tensors=[];colors=[]
    for yi in range(4):
        y1,y2=h*yi//4,h*(yi+1)//4
        for x1,x2 in xs:
            if x2<=x1 or y2<=y1:raise ValueError('Empty visible strip')
            rgb=np.asarray(image.crop((x1,y1,x2,y2)).convert('RGB'))
            lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32);lab[...,0]*=100/255;lab[...,1:]-=128
            colors.append(np.median(lab.reshape(-1,3),axis=0))
            crop=Image.fromarray(rgb).resize((224,224),Image.Resampling.BICUBIC)
            tensors.append(TF.normalize(TF.to_tensor(crop),MEAN,STD))
    vec=[]
    with torch.inference_mode():
        for start in range(0,8,4):
            vec.append(F.normalize(model.forward_features(torch.stack(tensors[start:start+4]).cuda())['x_norm_clstoken'],dim=1).cpu().numpy())
    return np.concatenate(vec),np.asarray(colors)


def rank(query_feat,query_lab,features,labs,names):
    cos=np.clip((1-np.sum(features*query_feat[None,:,:],axis=2))/2,0,1)
    col=np.clip(np.linalg.norm(labs-query_lab[None,:,:],axis=2)/100,0,1)
    value=(.8*cos+.2*col).mean(axis=1)
    return [names[i] for i in np.lexsort((names,value))[:5]]


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'examples').mkdir(exist_ok=True)
    validation=sorted((DATA/'validation/good').glob('*.png'))
    calibration=validation[:5];evaluation=validation[5:10]
    # Derived from normal design subset only, with fixed 8px 256-grid margin.
    extrema=[]
    for p in calibration:
        sig,err=signature(SEG/'validation/good'/p.name)
        if sig is None:raise RuntimeError('Design segmentation unavailable: '+p.name+' '+err)
        _,x=np.nonzero(sig['line']['mask']);extrema.extend([int(x.min()),int(x.max())+1])
    left=max(0,min(extrema)-8);right=min(256,max(extrema)+8)
    if left<=0 or right>=256:raise RuntimeError('Fixed-band design has no context on one side')
    print('band x',left,right,'on 256 grid; chosen on validation 000-004 only',flush=True)
    train=sorted((DATA/'train/good').glob('*.png'))
    if len(train)!=360:raise RuntimeError('Expected 360 reference images')
    names=np.asarray([p.stem for p in train]); hashes=[digest(p) for p in train];weight_hash=digest(WEIGHT)
    model=load_dinov3_model('dinov3_vitb16',pretrained_weight_path=str(WEIGHT)).cuda().eval()
    cache=OUT/'train_side_strips.npz'
    if cache.exists():
        with np.load(cache) as z:
            if list(z['names'])!=list(names) or list(z['hashes'])!=hashes or str(z['weight_hash'])!=weight_hash or list(z['band'])!=[left,right]:raise RuntimeError('Side-strip cache no longer matches its inputs')
            features,labs=z['features'],z['labs']
    else:
        ff=[];ll=[]
        for i,p in enumerate(train,1):
            with Image.open(p) as image: a,b=visible_strips(image.convert('RGB'),left,right,model)
            ff.append(a);ll.append(b)
            if i%50==0:print('training reference strips',i,'/',len(train),flush=True)
        features=np.stack(ff);labs=np.stack(ll);tmp=cache.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,names=names,hashes=hashes,weight_hash=weight_hash,band=[left,right],features=features,labs=labs);tmp.replace(cache)
    e3paths=sorted((CACHE/CAT/'train').glob('*.npz')); bank=[(p,read(p)) for p in e3paths]
    if len(bank)!=360 or config_hash(CACHE/'config.json')!=weight_hash:raise RuntimeError('E3 cache is incomplete or has another encoder')
    f16=np.stack([d['tile_features'] for _,d in bank]);lab16=np.stack([d['tile_lab'] for _,d in bank]);names16=np.asarray([p.stem for p,_ in bank])
    mean=f16.mean(axis=0); ordered=sorted((float(np.linalg.norm(d['tile_features']-mean)),p.stem) for p,d in bank)
    fixed=[ordered[i][1] for i in np.linspace(0,len(ordered)-1,5,dtype=int)]
    signatures={};invalid={}
    for name in names:
        sig,reason=signature(SEG/'train/good'/(name+'.png'));signatures[name]=sig
        if reason:invalid[reason]=invalid.get(reason,0)+1
    rows=[]; checks=[]; diagnostics=[]
    for i,path in enumerate(evaluation):
        with Image.open(path) as im: image=im.convert('RGB')
        donor_path=evaluation[(i+1)%len(evaluation)]
        with Image.open(donor_path) as im: donor=im.convert('RGB')
        if image.size!=donor.size:raise ValueError('Source/donor image sizes differ')
        feat,lab=visible_strips(image,left,right,model); bandrefs=rank(feat,lab,features,labs,names)
        q=read(CACHE/CAT/'validation'/(path.stem+'.npz'))
        if str(q['source_sha256'])!=digest(path):raise ValueError('Validation cache input changed')
        full=best_ids(tile_distances(q,f16,lab16),names16,list(range(16)))
        # Swap only the band; then independently re-encode the visible strips.
        x1=image.width*left//256;x2=image.width*right//256
        altered=image.copy();altered.paste(donor.crop((x1,0,x2,image.height)),(x1,0))
        changed=not np.array_equal(np.asarray(image.crop((x1,0,x2,image.height))),np.asarray(donor.crop((x1,0,x2,image.height))))
        feat2,lab2=visible_strips(altered,left,right,model);otherrefs=rank(feat2,lab2,features,labs,names)
        checks.append(dict(image=path.stem,donor=donor_path.stem,target_pixels_changed=changed,
            visible_feature_max_diff=float(np.max(abs(feat-feat2))),visible_lab_max_diff=float(np.max(abs(lab-lab2))),
            references_equal=bandrefs==otherrefs,reference_segment_hashes_equal=[digest(SEG/'train/good'/(x+'.png')) for x in bandrefs]==[digest(SEG/'train/good'/(x+'.png')) for x in otherrefs]))
        # Automatic segmentation is consulted only after references are fixed.
        obs,reason=signature(SEG/'validation/good'/path.name)
        if obs is None:
            diagnostics.append(dict(image=path.stem,status=reason));continue
        mask=obs['line']['mask']; outside=mask.copy();outside[:,left:right]=False
        visible_connector=[]
        for con in obs['connectors']:
            c=con['mask'];visible=c.copy();visible[:,left:right]=False
            visible_connector.append(float(visible.sum()/max(1,c.sum())))
        d=dict(image=path.stem,status='ok',line_pixels=int(mask.sum()),line_pixels_outside_band=int(outside.sum()),
               connector_visible_fraction=visible_connector)
        diagnostics.append(d)
        for method,ids in [('fixed',fixed),('full',full),('side_strips',bandrefs)]:
            usable=[signatures[x] for x in ids if signatures[x] is not None]
            same=[r for r in usable if r['line']['class_id']==obs['line']['class_id']]
            errs=[float(np.mean(abs(np.asarray(r['heights'])-obs['heights']))) for r in same]
            rows.append(dict(image=path.stem,method=method,reference_ids=';'.join(ids),
                usable_count=len(usable),same_color_count=len(same),endpoint_error_count=len(errs),
                mean_endpoint_error_same_color=float(np.mean(errs)) if errs else ''))
        if i==0:
            size=(512,256);original=image.resize(size);masked=original.copy()
            leftpx=size[0]*left//256;rightpx=size[0]*right//256
            ImageDraw.Draw(masked).rectangle((leftpx,0,rightpx-1,size[1]-1),fill=(90,90,90))
            canvas=Image.new('RGB',(1024,284),'white');canvas.paste(original,(0,28));canvas.paste(masked,(512,28))
            draw=ImageDraw.Draw(canvas);draw.text((5,4),'Normal '+path.stem,fill='black');draw.text((517,4),'Only side strips are encoded',fill='black')
            canvas.save(OUT/'examples'/(path.stem+'_band.png'))
    for filename,records in [('reference_results.csv',rows),('target_replacement.csv',checks)]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    summary=dict(protocol='exploratory fixed band determined on normal validation 000-004; check 005-009; line masks used only for design and offline evaluation, never query ranking',
        design_images=[p.name for p in calibration],evaluation_images=[p.name for p in evaluation],band_256_xyxy=[left,0,right,256],
        band_margin_pixels=8,training_count=len(train),invalid_training_signatures=invalid,fixed_references=fixed,
        diagnostics=diagnostics,replacement_checks=checks)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('replacement_checks',)},ensure_ascii=False,indent=2),flush=True)


def config_hash(path):
    return json.loads(path.read_text(encoding='utf-8'))['dino_sha256']


if __name__=='__main__':main()
