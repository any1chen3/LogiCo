"""Generate unlabeled SAM 2 proposals for later human E2 coverage audit."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'saved_results/external'
sys.path[:0] = [str(EXTERNAL / 'python_deps'), str(EXTERNAL / 'sam2')]
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

WEIGHT = ROOT / 'ckpt/sam2.1_hiera_tiny.pt'
OUT = ROOT / 'saved_results/context_e2_proposals'
EXPECTED_SHA = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
CATEGORIES = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']


def select_masks(proposals, height, width):
    filtered = []
    for index, proposal in enumerate(proposals):
        mask = proposal['segmentation'].astype(bool)
        area = int(mask.sum()) * (512 * 512) / (height * width)
        if 16 <= area <= .8 * 512 * 512:
            x, y, w, h = proposal['bbox']
            filtered.append((proposal, index, area, (int(x), int(y), int(w), int(h))))
    filtered.sort(key=lambda v: (-float(v[0]['predicted_iou']), -float(v[0]['stability_score']), v[3], v[1]))
    selected = []
    for proposal, index, area, bbox in filtered:
        m = proposal['segmentation']
        if any((np.logical_and(m, x[0]['segmentation']).sum() / np.logical_or(m, x[0]['segmentation']).sum()) > .85 for x in selected):
            continue
        selected.append((proposal, index, area, bbox))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--categories', nargs='+', default=CATEGORIES, choices=CATEGORIES)
    parser.add_argument('--points-per-batch', type=int, default=8)
    args = parser.parse_args()
    with WEIGHT.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != EXPECTED_SHA:
            raise ValueError('SAM 2 checkpoint SHA-256 mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('This audit is configured for GPU inference')
    OUT.mkdir(parents=True, exist_ok=True)
    model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', str(WEIGHT), device='cuda', apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(model, points_per_side=32, points_per_batch=args.points_per_batch,
                                           pred_iou_thresh=.80, stability_score_thresh=.90,
                                           box_nms_thresh=1.0, min_mask_region_area=0)
    records = []
    for category in args.categories:
        files = sorted((ROOT / 'data/mvtec_loco' / category / 'validation/good').glob('*.png'))[:args.limit]
        if len(files) < args.limit:
            raise RuntimeError(f'{category} only has {len(files)} validation images')
        for path in files:
            started = time.perf_counter()
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                original_size = rgb.size
                scale = 512 / max(rgb.size)
                rgb = rgb.resize((round(rgb.width*scale), round(rgb.height*scale)), Image.Resampling.BILINEAR)
                image = np.asarray(rgb)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                proposals = generator.generate(image)
            h, w = image.shape[:2]
            selected = select_masks(proposals, h, w)
            target = OUT / category / path.stem
            target.mkdir(parents=True, exist_ok=True)
            masks = np.stack([entry[0]['segmentation'] for entry in selected]) if selected else np.zeros((0, h, w), dtype=bool)
            np.savez_compressed(target / 'masks.npz', masks=masks)
            metadata = [dict(index=i, source_id=index, bbox_xywh=bbox, area_512=area,
                             predicted_iou=float(p['predicted_iou']), stability=float(p['stability_score']))
                        for i, (p, index, area, bbox) in enumerate(selected)]
            info = dict(category=category, image=str(path), source_size=list(original_size), mask_size=[w,h],
                        generated_after_sam_quality_filter=len(proposals), kept=len(selected), proposals=metadata)
            (target / 'metadata.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
            elapsed = time.perf_counter()-started
            row = dict(category=category, image=path.name, raw_after_quality=len(proposals), kept=len(selected),
                       seconds=round(elapsed,2), peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2,1),
                       manual_coverage='not_reviewed', manual_merge='not_reviewed', manual_duplicates='not_reviewed')
            records.append(row)
            print(json.dumps(row), flush=True)
    output = OUT / ('proposal_inventory_' + '_'.join(args.categories) + f'_{args.limit}.csv')
    with output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print('Saved', output, flush=True)


if __name__ == '__main__':
    main()
