"""Audit fixed irregular target masks designed on normal training-independent images."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/mvtec_loco/splicing_connectors/validation/good'
SEG = ROOT / 'data/mvtec_loco_segmentations/splicing_connectors/validation/good'
OUT = ROOT / 'saved_results/context_e4_fixed_shape_window_audit'
SIZE = (256, 256)


def read_labels(stem):
    with Image.open(SEG / f'{stem}.png') as image:
        labels = np.asarray(image.resize(SIZE, Image.Resampling.NEAREST))
    if labels.max() > 4:
        raise ValueError(f'Unexpected segmentation labels: {stem}')
    return labels


def read_rgb(stem):
    with Image.open(DATA / f'{stem}.png') as image:
        return np.asarray(image.convert('RGB').resize(SIZE, Image.Resampling.BILINEAR))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'examples').mkdir(exist_ok=True)
    design = [f'{i:03d}' for i in range(5)]
    evaluate = [f'{i:03d}' for i in range(10, 30)]
    union = np.zeros(SIZE, dtype=bool)
    for stem in design:
        union |= np.isin(read_labels(stem), (0, 1, 2))
    masks = {}
    for margin in (8, 16, 32):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        masks[f'shape_{margin}'] = cv2.dilate(union.astype('uint8'), kernel) > 0
    band = np.zeros(SIZE, dtype=bool)
    band[:, 55:207] = True
    masks['previous_band'] = band
    rows = []
    for name, mask in masks.items():
        Image.fromarray(mask.astype('uint8') * 255).save(OUT / f'{name}_mask.png')
        for index, stem in enumerate(evaluate):
            labels = read_labels(stem)
            line = np.isin(labels, (0, 1, 2))
            number, components, stats, _ = cv2.connectedComponentsWithStats((labels == 3).astype('uint8'), 8)
            connectors = [(stats[i, 0], components == i) for i in range(1, number) if stats[i, 4] >= 4]
            connectors.sort(key=lambda item: item[0])
            fractions = [float((component & ~mask).sum() / component.sum()) for _, component in connectors]
            outside = int((line & ~mask).sum())
            original = read_rgb(stem)
            donor = read_rgb(evaluate[(index + 1) % len(evaluate)])
            replaced = original.copy()
            replaced[mask] = donor[mask]
            visible_before = original.copy(); visible_before[mask] = 90
            visible_after = replaced.copy(); visible_after[mask] = 90
            changed = bool(np.any(original[mask] != replaced[mask]))
            same_input = bool(np.array_equal(visible_before, visible_after))
            rows.append(dict(image=stem, method=name, mask_fraction=float(mask.mean()), line_pixels=int(line.sum()),
                line_pixels_outside=outside, line_fully_hidden=bool(line.any() and outside == 0),
                connector_count=len(connectors), left_visible=fractions[0] if len(connectors) == 2 else '',
                right_visible=fractions[1] if len(connectors) == 2 else '',
                hidden_and_both_80=bool(line.any() and outside == 0 and len(fractions) == 2 and min(fractions) >= .8),
                donor=evaluate[(index + 1) % len(evaluate)], target_pixels_changed=changed, model_input_identical=same_input))
            if stem in ('010', '014', '020'):
                Image.fromarray(visible_before).resize((512, 256), Image.Resampling.NEAREST).save(OUT / 'examples' / f'{stem}_{name}.png')
    with (OUT / 'per_image.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    result = {}
    for name, mask in masks.items():
        group = [r for r in rows if r['method'] == name]
        valid = [r for r in group if r['connector_count'] == 2]
        result[name] = dict(mask_fraction=float(mask.mean()), images=len(group),
            line_fully_hidden=sum(r['line_fully_hidden'] for r in group),
            mean_line_fraction_hidden=float(np.mean([1-r['line_pixels_outside']/r['line_pixels'] for r in group if r['line_pixels']])),
            connector_valid=len(valid), left_visible_mean=float(np.mean([r['left_visible'] for r in valid])),
            right_visible_mean=float(np.mean([r['right_visible'] for r in valid])),
            hidden_and_both_80=sum(r['hidden_and_both_80'] for r in group),
            target_changed=sum(r['target_pixels_changed'] for r in group),
            model_input_identical=sum(r['model_input_identical'] for r in group))
    summary = dict(protocol='Fixed mask from union of line segmentation on normal validation 000-004, dilated with 8/16/32 px radius. Same masks for all normal images 010-029. Images 010-019 previously seen in thumbnail review. Automatic segmentation is diagnostic only on evaluated images; no anomalous data.',
        replacement='Replace pixels inside fixed mask with another normal image, then fill the same fixed mask before any encoding; only raw pre-encoder equality checked, no actual features or rankings computed.',
        design=design, evaluation=evaluate, result=result)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
