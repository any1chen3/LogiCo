"""Diagnostic comparison to LogiCo's automatic class maps, not human GT."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import yaml
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
PROPOSALS = ROOT / 'saved_results/context_e2_proposals'
SEGMENTS = ROOT / 'data/mvtec_loco_segmentations'
OUT = ROOT / 'saved_results/context_e2_existing_seg_comparison'
BACKGROUND = {'black background', 'grid'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, parts, missing = [], [], []
    for file in sorted(PROPOSALS.glob('*/*/metadata.json')):
        info = json.loads(file.read_text(encoding='utf-8'))
        category = info['category']
        stem = Path(info['image']).stem
        ref_path = SEGMENTS / category / 'validation/good' / (stem + '.png')
        if not ref_path.exists():
            missing.append(str(ref_path))
            continue
        prompts = yaml.safe_load((ROOT / 'configs/segmentations/mvtec_loco' / (category + '.yaml')).read_text(encoding='utf-8'))['prompts']
        with Image.open(ref_path) as im:
            if im.mode not in ('P', 'L'):
                raise ValueError('Expected integer label map: ' + str(ref_path))
            labels = np.asarray(im.resize(tuple(info['mask_size']), Image.Resampling.NEAREST))
        with np.load(file.parent / 'masks.npz') as npz:
            masks = npz['masks']
        if labels.size and int(labels.max()) >= len(prompts):
            raise ValueError('Unexpected class ID: ' + str(ref_path))
        for idx, name in enumerate(prompts):
            if name.lower() in BACKGROUND:
                continue
            ref = labels == idx
            area = int(ref.sum())
            if area == 0:
                continue
            intersection = np.logical_and(masks, ref).sum(axis=(1, 2))
            union = np.logical_or(masks, ref).sum(axis=(1, 2))
            iou = intersection / np.maximum(union, 1)
            best = int(np.argmax(iou)) if len(iou) else -1
            rows.append(dict(category=category, image=stem, class_id=idx, class_name=name, reference_pixels=area,
                             best_candidate=best, best_iou=float(iou[best]) if best >= 0 else 0.,
                             recall_at_best_iou=float(intersection[best]/area) if best >= 0 else 0.,
                             passes_iou_0_5=bool(best >= 0 and iou[best] >= .5)))
            components, count = label(ref, structure=np.ones((3, 3)))
            for component_id in range(1, count + 1):
                component = components == component_id
                pixels = int(component.sum())
                normalized_area = pixels * 512 * 512 / labels.size
                if normalized_area < 16:
                    continue
                intersections = np.logical_and(masks, component).sum(axis=(1, 2))
                unions = np.logical_or(masks, component).sum(axis=(1, 2))
                values = intersections / np.maximum(unions, 1)
                candidate_id = int(np.argmax(values)) if len(values) else -1
                parts.append(dict(category=category, image=stem, class_id=idx, class_name=name,
                                  connected_region=component_id, reference_pixels=pixels, best_candidate=candidate_id,
                                  best_iou=float(values[candidate_id]) if candidate_id >= 0 else 0.,
                                  passes_iou_0_5=bool(candidate_id >= 0 and values[candidate_id] >= .5)))
    fields = ['category','image','class_id','class_name','reference_pixels','best_candidate','best_iou','recall_at_best_iou','passes_iou_0_5']
    with (OUT/'class_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    part_fields = ['category','image','class_id','class_name','connected_region','reference_pixels','best_candidate','best_iou','passes_iou_0_5']
    with (OUT/'connected_region_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=part_fields); writer.writeheader(); writer.writerows(parts)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['category']].append(row)
    summary = {}
    for category, subset in grouped.items():
        summary[category] = dict(images=len({r['image'] for r in subset}), present_class_regions=len(subset),
                                 fraction_best_iou_at_least_0_5=sum(r['passes_iou_0_5'] for r in subset)/len(subset),
                                 mean_best_iou=float(np.mean([r['best_iou'] for r in subset])))
        category_parts = [r for r in parts if r['category'] == category]
        summary[category]['connected_regions'] = len(category_parts)
        summary[category]['connected_fraction_iou_at_least_0_5'] = sum(r['passes_iou_0_5'] for r in category_parts) / len(category_parts) if category_parts else None
        cases = sorted(subset, key=lambda r: (r['best_iou'], r['image'], r['class_id']))[:5]
        for rank, row in enumerate(cases, 1):
            info = json.loads((PROPOSALS/category/row['image']/'metadata.json').read_text(encoding='utf-8'))
            with Image.open(info['image']) as image:
                base = image.convert('RGB').resize(tuple(info['mask_size']), Image.Resampling.BILINEAR)
            with Image.open(SEGMENTS/category/'validation/good'/(row['image']+'.png')) as image:
                labels = np.asarray(image.resize(base.size, Image.Resampling.NEAREST))
            with np.load(PROPOSALS/category/row['image']/'masks.npz') as archive:
                candidate = archive['masks'][row['best_candidate']] if row['best_candidate'] >= 0 else np.zeros(labels.shape, bool)
            panels = [base]
            for mask, color in [(labels == row['class_id'], (255, 40, 40)), (candidate, (30, 130, 255))]:
                panel = base.copy()
                panel.paste(color, (0, 0, *base.size), Image.fromarray(mask.astype(np.uint8)*120))
                panels.append(panel)
            w, h = base.size
            canvas = Image.new('RGB', (3*w, h+40), 'white')
            for j, panel in enumerate(panels): canvas.paste(panel, (j*w, 40))
            ImageDraw.Draw(canvas).text((5, 5), f"{category} {row['image']} {row['class_name']} IoU={row['best_iou']:.3f} | original | existing (red) | SAM best (blue)", fill='black')
            dest = OUT/'cases'/category
            dest.mkdir(parents=True, exist_ok=True)
            canvas.save(dest/f"{rank:02d}_{row['image']}_class{row['class_id']}.jpg", quality=90)
    result = dict(reference='existing automatically generated text-prompt class segmentation; not human GT',
                  unit='one present non-background class region per image, may merge multiple instances; connected regions are not guaranteed to be real instances',
                  threshold=.5, excluded_background=sorted(BACKGROUND), missing=missing, by_category=summary)
    (OUT/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit('Missing segmentation files')


if __name__ == '__main__':
    main()
