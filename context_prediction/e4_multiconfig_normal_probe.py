"""Normal-only diagnostic of multi-configuration reference-line matching."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from context_prediction.e4_full_line_mask_probe import DATA, SEG, signature
from context_prediction.e4_normal_alignment_probe import connector_transform, compare

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'saved_results/context_e4_connector_context_validation/per_image.csv'
OUT=ROOT/'saved_results/context_e4_multiconfig_normal_probe'
# Exploratory proxy only; no tuning on anomalies or claiming a physical match.
MATCH_IOU=.10


def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'examples').mkdir(exist_ok=True)
    with INPUT.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    queries=[r for r in rows if r['method']=='side_strips' and r['status']=='ok' and r['line_pixels_outside']=='0']
    if len(queries)!=23: raise ValueError('Expected 23 previously audited normal queries')
    results=[];checks=[]
    for q in queries:
        stem=q['image']; observed,reason=signature(SEG/'validation/good'/(stem+'.png'))
        if observed is None: raise ValueError(stem+': '+reason)
        obs=observed['line']['mask']; masks=[]; residuals=[]; matches=[]; ids=[]; mismatch=[]
        for refid in q['reference_ids'].split(';'):
            ref,err=signature(SEG/'train/good'/(refid+'.png'))
            if ref is None: mismatch.append(refid+':'+err);continue
            transform=connector_transform(ref['connectors'],observed['connectors'])
            if transform is None: mismatch.append(refid+':invalid_transform');continue
            m,_,_=transform
            target=cv2.warpAffine(ref['line']['mask'].astype('uint8'),m,(256,256),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
            if not target.any(): mismatch.append(refid+':empty_after_alignment');continue
            iou,_,_=compare(target,obs)
            dilated=cv2.dilate(obs.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
            missing=target & ~dilated
            same=ref['line']['class_id']==observed['line']['class_id']
            match=bool(same and iou>=MATCH_IOU)
            ids.append(refid); masks.append(target); residuals.append(missing);matches.append(match)
            checks.append(dict(image=stem,reference=refid,same_class=bool(same),aligned_iou=iou,matched=match))
        if not masks:continue
        masks=np.stack(masks);residuals=np.stack(residuals); selected=np.asarray(matches,bool)
        baseline=residuals.mean(axis=0)
        # Alternative hypotheses: minimum among references already matched to the real line.
        # If none match, fall back to baseline rather than dropping all evidence.
        revised=residuals[selected].min(axis=0).astype(float) if selected.any() else baseline.copy()
        # Show the dangerous universal intersection separately for empty-evidence diagnosis.
        naive=residuals.min(axis=0).astype(float)
        support=masks.any(axis=0); den=max(1,int(support.sum()))
        def total(x):return float(x.sum())
        results.append(dict(image=stem,usable_refs=len(ids),unusable_refs=';'.join(mismatch),matched_refs=int(selected.sum()),
            matched_reference_ids=';'.join(np.asarray(ids)[selected]),mean_reference_residual_mass=total(baseline),
            multiconfig_residual_mass=total(revised),naive_intersection_mass=total(naive),
            support_pixels=int(support.sum()),baseline_per_support=total(baseline)/den,
            multiconfig_per_support=total(revised)/den,naive_per_support=total(naive)/den,
            used_fallback=not bool(selected.any()),baseline_nonzero_pixels=int((baseline>0).sum()),
            multiconfig_nonzero_pixels=int((revised>0).sum())))
        with Image.open(DATA/'validation/good'/(stem+'.png')) as im: orig=im.convert('RGB').resize((512,256))
        def mask_img(v):return Image.fromarray(np.uint8(v)*255).convert('RGB').resize((512,256),Image.Resampling.NEAREST)
        def heat(v):
            colors=cv2.applyColorMap(np.uint8(np.clip(v,0,1)*255),cv2.COLORMAP_INFERNO)
            return Image.fromarray(cv2.cvtColor(colors,cv2.COLOR_BGR2RGB)).resize((512,256),Image.Resampling.NEAREST)
        panels=[('Normal '+stem,orig),('Observed line',mask_img(obs)),('References occupancy (mean)',heat(masks.mean(axis=0))),
                ('Mean residual (baseline)',heat(baseline)),('Matched alternatives / fallback',heat(revised)),('Naive intersection (warning)',heat(naive))]
        canvas=Image.new('RGB',(1536,572),'white');draw=ImageDraw.Draw(canvas)
        for i,(title,panel) in enumerate(panels):
            x=(i%3)*512;y=(i//3)*286;canvas.paste(panel,(x,y+30));draw.text((x+4,y+7),title,fill='black')
        canvas.save(OUT/'examples'/f'{stem}.jpg',quality=90)
    for file,items in [('per_image.csv',results),('reference_matching.csv',checks)]:
        with (OUT/file).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=items[0]);w.writeheader();w.writerows(items)
    withmatch=[r for r in results if r['matched_refs']>0]; none=[r for r in results if not r['matched_refs']]
    summary=dict(protocol='Normal-only exploratory alternative-reference diagnostic on 23 fixed validation queries and their previously selected reference IDs. Retrieval unchanged. Post-hoc matching uses automatic class and aligned line mask IoU >= 0.10; alignment uses complete connector masks. No anomaly ground truth, training, or sPRO.',
        match_definition='A reference is called matched if the existing auto class agrees and full-mask aligned IoU >= 0.10; this is an exploratory threshold, not proof of physical colour or correct match.',
        calculation='baseline = mean of per-reference missing masks after 2px observed dilation; proposed = per-pixel minimum among matched reference missing masks, fallback to baseline when no matched references; naive = per-pixel minimum across all references for failure demonstration.',
        caution='Uses observed line masks after retrieval; hypothetical missing line has no observed match. Same-class alternatives may be mutually exclusive configurations. Residual masses use only valid reference masks, not calibrated FPR. Do not treat as deployable or anomaly detection performance.',
        images=len(results),with_matched=len(withmatch),without_matched=len(none),images_without_match=[r['image'] for r in none],
        mean_baseline_mass=float(np.mean([r['mean_reference_residual_mass'] for r in results])),
        mean_multiconfig_mass=float(np.mean([r['multiconfig_residual_mass'] for r in results])),
        mean_naive_mass=float(np.mean([r['naive_intersection_mass'] for r in results])),
        conditional_improved=sum(r['multiconfig_residual_mass']<r['mean_reference_residual_mass'] for r in results),
        naive_zero=sum(r['naive_intersection_mass']==0 for r in results),
        case_013=next(r for r in results if r['image']=='013'))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
