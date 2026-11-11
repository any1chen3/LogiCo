"""Exploratory oracle union against existing segmentations, never a predictor."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'saved_results/context_e2_existing_seg_comparison'
OUT = ROOT / 'saved_results/context_e2_union_diagnosis'
PROPS = ROOT / 'saved_results/context_e2_proposals'
SEGS = ROOT / 'data/mvtec_loco_segmentations'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (BASE/'class_comparison.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['category'] in ('splicing_connectors','juice_bottle')]
    output = []
    for row in rows:
        category, stem = row['category'], row['image']
        path = PROPS/category/stem
        info = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        with Image.open(SEGS/category/'validation/good'/(stem+'.png')) as im:
            ref = np.asarray(im.resize(tuple(info['mask_size']), Image.Resampling.NEAREST)) == int(row['class_id'])
        with np.load(path/'masks.npz') as archive:
            masks = archive['masks']
        union = np.zeros(ref.shape, dtype=bool)
        selected, score = [], 0.
        for _ in range(5):
            best_id, best_score = None, score
            for i, mask in enumerate(masks):
                if i in selected: continue
                merged = np.logical_or(union, mask)
                overlap = np.logical_and(merged, ref).sum()
                iou = overlap / max(1, np.logical_or(merged, ref).sum())
                if iou > best_score + 1e-12:
                    best_id, best_score = i, float(iou)
            if best_id is None: break
            union |= masks[best_id]
            selected.append(best_id)
            score = best_score
        output.append(dict(category=category, image=stem, class_id=row['class_id'], class_name=row['class_name'],
                           single_iou=float(row['best_iou']), union_iou=score, delta=score-float(row['best_iou']),
                           union_reference_recall=float(np.logical_and(union, ref).sum()/ref.sum()),
                           selected_ids=','.join(map(str, selected))))
    with (OUT/'oracle_union_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=output[0].keys()); writer.writeheader(); writer.writerows(output)
    for category in ('splicing_connectors','juice_bottle'):
        subset=[r for r in output if r['category']==category]
        cases=sorted(subset,key=lambda r:(-r['delta'],r['image'],r['class_id']))[:3] + sorted(subset,key=lambda r:(r['union_iou'],r['image'],r['class_id']))[:3]
        for rank,row in enumerate(cases,1):
            path=PROPS/category/row['image']
            info=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
            with Image.open(info['image']) as im: base=im.convert('RGB').resize(tuple(info['mask_size']),Image.Resampling.BILINEAR)
            with Image.open(SEGS/category/'validation/good'/(row['image']+'.png')) as im:
                ref=np.asarray(im.resize(base.size,Image.Resampling.NEAREST))==int(row['class_id'])
            with np.load(path/'masks.npz') as archive:
                masks=archive['masks']
                selected=[int(x) for x in row['selected_ids'].split(',') if x]
                union=np.any(masks[selected],axis=0) if selected else np.zeros(ref.shape,bool)
            w,h=base.size; canvas=Image.new('RGB',(w*3,h+40),'white'); canvas.paste(base,(0,40))
            for j,(mask,color) in enumerate([(ref,(255,40,40)),(union,(30,130,255))],1):
                panel=base.copy(); panel.paste(color,(0,0,w,h),Image.fromarray(mask.astype(np.uint8)*120)); canvas.paste(panel,(w*j,40))
            ImageDraw.Draw(canvas).text((3,5),f"{category} {row['image']} {row['class_name']}: {row['single_iou']:.3f} -> {row['union_iou']:.3f}; selected {row['selected_ids']}",fill='black')
            dest=OUT/'cases'/category; dest.mkdir(parents=True,exist_ok=True); canvas.save(dest/f"{rank:02d}_{row['image']}_class{row['class_id']}.jpg",quality=90)
    summary={}
    for category in ('splicing_connectors','juice_bottle'):
        subset=[r for r in output if r['category']==category]
        summary[category]=dict(regions=len(subset),single_pass=sum(r['single_iou']>=.5 for r in subset),union_pass=sum(r['union_iou']>=.5 for r in subset),mean_gain=float(np.mean([r['delta'] for r in subset])))
    (OUT/'summary.json').write_text(json.dumps(dict(protocol='exploratory greedy selection against reference, maximum 5 masks; oracle, not inference or human GT',categories=summary),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
