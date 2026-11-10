"""Evaluate proposals against independently supplied human reference masks.

Expected layout: reference_root/category/image_stem/component_name.png. Each
PNG is a binary, human-reviewed mask at the source image resolution. An absent
reference is omitted and reported, never counted as successful coverage.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'saved_results/context_e2_proposals'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, no_reference = [], []
    for meta in sorted(DATA.glob('*/*/metadata.json')):
        info = json.loads(meta.read_text(encoding='utf-8'))
        ref_dir = args.references / info['category'] / Path(info['image']).stem
        reference_files = sorted(ref_dir.glob('*.png')) if ref_dir.exists() else []
        if not reference_files:
            no_reference.append(str(ref_dir))
            continue
        with np.load(meta.parent / 'masks.npz') as archive:
            candidates = archive['masks']
        w, h = info['mask_size']
        for path in reference_files:
            with Image.open(path) as img:
                if img.size != tuple(info['source_size']):
                    raise ValueError(f'wrong source size: {path}')
                ref = np.asarray(img.convert('L').resize((w,h), Image.Resampling.NEAREST)) > 0
            if not ref.any():
                raise ValueError(f'empty reference: {path}')
            intersection = np.logical_and(candidates, ref).sum(axis=(1,2))
            union = np.logical_or(candidates, ref).sum(axis=(1,2))
            iou = intersection / np.maximum(union, 1)
            best = int(np.argmax(iou)) if len(iou) else None
            rows.append(dict(category=info['category'], image=Path(info['image']).name, component=path.stem,
                             area_512=float(ref.sum()*512*512/(w*h)), best_candidate=best,
                             best_iou=float(iou[best]) if best is not None else 0.,
                             best_component_recall=float(intersection[best]/ref.sum()) if best is not None else 0.,
                             covered_iou_0_5=bool(best is not None and iou[best] >= .5)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['category','image','component','area_512','best_candidate','best_iou','best_component_recall','covered_iou_0_5']
    with args.output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(reference_components=len(rows), images_without_reference=len(no_reference),
                   missing_reference_directories=no_reference,
                   note='Human reference masks required; this file does not assess merges or duplicates.')
    args.output.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'missing_reference_directories'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
