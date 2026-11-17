"""Normal-only diagnostic: fixed line pairs, connector-derived similarity transform."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/mvtec_loco/splicing_connectors'
SEG=ROOT/'data/mvtec_loco_segmentations/splicing_connectors'
PRIOR=ROOT/'saved_results/context_e4_normal_line_audit/line_pairs.csv'
OUT=ROOT/'saved_results/context_e4_normal_alignment_probe'


def read_components(path):
    with Image.open(path) as img: labels=np.asarray(img.resize((256,256),Image.Resampling.NEAREST))
    result={}
    for cls in range(4):
        count,components,stats,centers=cv2.connectedComponentsWithStats(np.uint8(labels==cls),8)
        for idx in range(1,count):
            x,y,w,h,area=stats[idx]
            if area < 4: continue
            result[f'{cls}:{idx}']=dict(mask=components==idx,center=centers[idx]+.5,box=(x,y,w,h),area=int(area),class_id=cls)
    return result


def anchors(components):
    conn=[v for v in components.values() if v['class_id']==3]
    if len(conn)!=2:return None
    return sorted(conn,key=lambda z:z['center'][0])


def connector_transform(reference,observed):
    """Fit a 2-point similarity in 512x256 aspect-aware coordinates."""
    left,right=reference; left2,right2=observed
    p=np.array([left['center']*[2,1],right['center']*[2,1]],float)
    q=np.array([left2['center']*[2,1],right2['center']*[2,1]],float)
    v=p[1]-p[0];u=q[1]-q[0]; denom=v@v
    if denom<1:return None
    a=(v@u)/denom;b=(v[0]*u[1]-v[1]*u[0])/denom
    m=np.array([[a,-b],[b,a]]); t=q[0]-m@p[0]
    aspect=np.diag([2.,1.]); linear=np.linalg.inv(aspect)@m@aspect; shift=np.linalg.inv(aspect)@t
    # Connector centroids use pixel centers (index + 0.5), while warpAffine
    # operates in zero-based pixel-index coordinates.
    shift=shift+linear@np.array([.5,.5])-.5
    return np.column_stack((linear,shift)),float(np.hypot(a,b)),float(np.degrees(np.arctan2(b,a)))


def end_height(line,anchors_pair):
    yy,xx=np.nonzero(line)
    if len(xx)<5:return None
    left,right=np.quantile(xx,[.15,.85]); result=[]
    for selector,connector in [(xx<=left,anchors_pair[0]),(xx>=right,anchors_pair[1])]:
        x,y,w,h=connector['box']
        if not selector.any() or h<1:return None
        result.append(float((np.median(yy[selector])+0.5-y)/h))
    return result


def compare(reference,observed):
    i=np.logical_and(reference,observed).sum();u=np.logical_or(reference,observed).sum()
    inflated=cv2.dilate(observed.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    missing=reference&~inflated
    return float(i/max(1,u)),float(missing.sum()/max(1,reference.sum())),missing


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'examples').mkdir(exist_ok=True)
    with PRIOR.open(encoding='utf-8-sig',newline='') as f:
        allrows=[r for r in csv.DictReader(f) if r['status']=='matched']
    unique={}
    for row in allrows:
        key=(row['image'],row['reference'],row['expected_id'],row['observed_id'])
        unique.setdefault(key,row)
    segcache={}; results=[]; skipped=[]
    for key,row in sorted(unique.items()):
        image,reference,expected_id,observed_id=key
        def get(split,stem):
            k=(split,stem)
            if k not in segcache: segcache[k]=read_components(SEG/split/'good'/(stem+'.png'))
            return segcache[k]
        obs=get('validation',image);ref=get('train',reference)
        obs_anchor,ref_anchor=anchors(obs),anchors(ref)
        if expected_id not in ref or observed_id not in obs or obs_anchor is None or ref_anchor is None:
            skipped.append(dict(image=image,reference=reference,reason='missing matched component or not exactly two connector regions'))
            continue
        result=connector_transform(ref_anchor,obs_anchor)
        if result is None:
            skipped.append(dict(image=image,reference=reference,reason='degenerate connector geometry'));continue
        matrix,scale,angle=result
        refmask=ref[expected_id]['mask'];obsmask=obs[observed_id]['mask']
        aligned=cv2.warpAffine(refmask.astype('uint8'),matrix,(256,256),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
        before_iou,before_residual,before_missing=compare(refmask,obsmask)
        after_iou,after_residual,after_missing=compare(aligned,obsmask)
        ref_end=end_height(refmask,ref_anchor);obs_end=end_height(obsmask,obs_anchor)
        row=dict(image=image,reference=reference,expected_id=expected_id,observed_id=observed_id,
                 scale=scale,angle_degrees=angle,source_pixels=int(refmask.sum()),aligned_pixels=int(aligned.sum()),
                 iou_before=before_iou,iou_after=after_iou,residual_before=before_residual,residual_after=after_residual,
                 left_reference_height=ref_end[0] if ref_end else '',right_reference_height=ref_end[1] if ref_end else '',
                 left_observed_height=obs_end[0] if obs_end else '',right_observed_height=obs_end[1] if obs_end else '',
                 mean_endpoint_height_difference=float(np.mean(np.abs(np.array(ref_end)-obs_end))) if ref_end and obs_end else '')
        results.append(row)
        if image=='000' and reference=='330':
            with Image.open(DATA/'validation/good'/(image+'.png')) as im: background=im.convert('RGB').resize((512,256))
            def paint(ref_line):
                red=np.uint8(ref_line)*255;green=np.uint8(obsmask)*255
                rgb=np.stack([red,green,np.zeros_like(red)],axis=-1)
                return Image.blend(background,Image.fromarray(rgb).resize((512,256),Image.Resampling.NEAREST),.55)
            out=Image.new('RGB',(1536,284),'white'); draw=ImageDraw.Draw(out)
            for i,(panel,title) in enumerate([(background,'Normal 000'),(paint(refmask),'Before: ref 330 red / observed green'),(paint(aligned),'After connector alignment')]):
                out.paste(panel,(512*i,28));draw.text((512*i+5,4),title,fill='black')
            out.save(OUT/'examples/000_reference330.png')
    if not results:raise RuntimeError('No usable line pairs')
    with (OUT/'paired_lines.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
    per_image=[]
    for image in sorted({x['image'] for x in results}):
        r=[x for x in results if x['image']==image]
        med=lambda k:float(np.median([x[k] for x in r]))
        per_image.append(dict(image=image,unique_pairs=len(r),median_iou_before=med('iou_before'),median_iou_after=med('iou_after'),
                              median_residual_before=med('residual_before'),median_residual_after=med('residual_after'),
                              median_endpoint_difference=float(np.median([x['mean_endpoint_height_difference'] for x in r if x['mean_endpoint_height_difference']!=''])) if any(x['mean_endpoint_height_difference']!='' for x in r) else None))
    with (OUT/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=per_image[0]);w.writeheader();w.writerows(per_image)
    summary=dict(description='Exploratory on normal validation 000-009; unique matched components from previously fixed pairs; registration only uses two original segmentation connector centroids; no matching or retrieval update',
                 matched_window_records=len(allrows),unique_input_pairs=len(unique),evaluated_pairs=len(results),skipped=skipped,per_image=per_image,
                 median_per_image_iou_before=float(np.median([x['median_iou_before'] for x in per_image])),median_per_image_iou_after=float(np.median([x['median_iou_after'] for x in per_image])),
                 median_per_image_residual_before=float(np.median([x['median_residual_before'] for x in per_image])),median_per_image_residual_after=float(np.median([x['median_residual_after'] for x in per_image])))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
