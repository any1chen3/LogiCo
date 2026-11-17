"""Diagnose line shape of previously retrieved references on normal images."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from context_prediction.e4_full_line_mask_probe import DATA, SEG, signature
from context_prediction.e4_normal_alignment_probe import connector_transform, compare

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'saved_results/context_e4_connector_context_validation/per_image.csv'
OUT = ROOT / 'saved_results/context_e4_reference_line_diagnosis'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'examples').mkdir(exist_ok=True)
    with INPUT.open(encoding='utf-8-sig', newline='') as f: rows=list(csv.DictReader(f))
    queries=[r for r in rows if r['method']=='side_strips' and r['status']=='ok' and r['line_pixels_outside']=='0']
    if len(queries)!=23: raise ValueError('Expected the same 23 normal queries')
    pairs=[]; skips=[]
    for row in queries:
        stem=row['image']; obs,reason=signature(SEG/'validation/good'/(stem+'.png'))
        if obs is None: raise ValueError('Query signature changed '+stem+' '+reason)
        with Image.open(DATA/'validation/good'/(stem+'.png')) as im: background=im.convert('RGB').resize((512,256))
        observed=obs['line']['mask']
        for refid in row['reference_ids'].split(';'):
            ref,reason=signature(SEG/'train/good'/(refid+'.png'))
            if ref is None:
                skips.append(dict(image=stem,reference=refid,reason=reason));continue
            transformation=connector_transform(ref['connectors'],obs['connectors'])
            if transformation is None:
                skips.append(dict(image=stem,reference=refid,reason='invalid_transform'));continue
            matrix,scale,angle=transformation
            expected=ref['line']['mask']
            aligned=cv2.warpAffine(expected.astype('uint8'),matrix,(256,256),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
            raw_iou,raw_residual,_=compare(expected,observed)
            aligned_iou,aligned_residual,_=compare(aligned,observed)
            if not aligned.any():
                skips.append(dict(image=stem,reference=refid,reason='empty_after_alignment'));continue
            same=obs['line']['class_id']==ref['line']['class_id']
            pairs.append(dict(image=stem,reference=refid,query_class=obs['line']['class_id'],reference_class=ref['line']['class_id'],same_class=bool(same),
                raw_iou=raw_iou,aligned_iou=aligned_iou,raw_reference_residual=raw_residual,aligned_reference_residual=aligned_residual,
                reference_pixels=int(expected.sum()),aligned_pixels=int(aligned.sum()),observed_pixels=int(observed.sum()),scale=scale,angle_degrees=angle))
            def overlay(a):
                rgb=np.stack((np.uint8(a)*255,np.uint8(observed)*255,np.zeros((256,256),dtype='uint8')),axis=-1)
                return Image.blend(background,Image.fromarray(rgb).resize((512,256),Image.Resampling.NEAREST),.55)
            with Image.open(DATA/'train/good'/(refid+'.png')) as im: reference=im.convert('RGB').resize((512,256))
            panels=[('Normal '+stem,background),('Reference '+refid,reference),('Raw red ref / green actual',overlay(expected)),('Aligned red ref / green actual',overlay(aligned))]
            canvas=Image.new('RGB',(2048,286),'white');draw=ImageDraw.Draw(canvas)
            for i,(title,panel) in enumerate(panels):
                canvas.paste(panel,(i*512,30));draw.text((i*512+5,6),title,fill='black')
            canvas.save(OUT/'examples'/f'{stem}_ref_{refid}.jpg',quality=88)
    if not pairs:raise ValueError('No pairs')
    with (OUT/'reference_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=pairs[0]);writer.writeheader();writer.writerows(pairs)
    per=[]
    for q in queries:
        group=[p for p in pairs if p['image']==q['image']]
        if not group:continue
        average=lambda key: float(np.mean([p[key] for p in group]))
        per.append(dict(image=q['image'],usable=len(group),same_class=sum(p['same_class'] for p in group),
            mean_raw_iou=average('raw_iou'),mean_aligned_iou=average('aligned_iou'),
            mean_raw_residual=average('raw_reference_residual'),mean_aligned_residual=average('aligned_reference_residual'),
            best_aligned_iou_reference=max(group,key=lambda p:p['aligned_iou'])['reference'],
            worst_aligned_iou_reference=min(group,key=lambda p:p['aligned_iou'])['reference']))
    with (OUT/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=per[0]);writer.writeheader();writer.writerows(per)
    mean=lambda key:float(np.mean([p[key] for p in per]))
    summary=dict(protocol='23 normal validation queries, the five reference IDs selected by the previous fixed-band side-context retrieval, no changes to selection. Two-connector similarity alignment from full automatic connector masks is retrospective diagnosis only, not available as validated leak-free model.',
        caveats='Automatic segmentation not human truth; IoU penalizes width and thin shifts, and residual is directed reference minus 2px-dilated observed divided by reference area; alignment/interpolation can clip pixels. These are not sPRO and cannot alone establish which error source caused each failure.',
        query_count=len(queries),evaluated_queries=len(per),expected_pairs=len(queries)*5,usable_pairs=len(pairs),skipped=skips,
        mean_per_image_raw_iou=mean('mean_raw_iou'),mean_per_image_aligned_iou=mean('mean_aligned_iou'),
        mean_per_image_raw_residual=mean('mean_raw_residual'),mean_per_image_aligned_residual=mean('mean_aligned_residual'),
        improved_iou_images=sum(p['mean_aligned_iou']>p['mean_raw_iou'] for p in per),
        decreased_residual_images=sum(p['mean_aligned_residual']<p['mean_raw_residual'] for p in per),
        worst_images_by_aligned_iou=sorted(per,key=lambda x:x['mean_aligned_iou'])[:5])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
