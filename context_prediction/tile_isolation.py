"""Preliminary real-image target replacement check, not the full E3 evaluation."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from dinov3.hub.backbones import load_dinov3_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'saved_results/context_e3_preliminary'
WEIGHT = ROOT / 'ckpt/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def boxes(width, height):
    xs = [width * i // 4 for i in range(5)]
    ys = [height * i // 4 for i in range(5)]
    return [(xs[x], ys[y], xs[x+1], ys[y+1]) for y in range(4) for x in range(4)]


def encode_tiles(image, model, device):
    rgb = np.asarray(image.convert('RGB'))
    tiles, colors = [], []
    for x1, y1, x2, y2 in boxes(image.width, image.height):
        tile = rgb[y1:y2, x1:x2]
        lab = cv2.cvtColor(tile, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 0] *= 100 / 255
        lab[..., 1:] -= 128
        colors.append(np.median(lab.reshape(-1, 3), axis=0))
        crop = Image.fromarray(tile).resize((224, 224), Image.Resampling.BICUBIC)
        tiles.append(TF.normalize(TF.to_tensor(crop), MEAN, STD))
    embeddings = []
    with torch.inference_mode():
        for batch in torch.stack(tiles).split(4):
            cls = model.forward_features(batch.to(device))['x_norm_clstoken']
            embeddings.append(F.normalize(cls, dim=1).cpu().numpy())
    return np.concatenate(embeddings), np.asarray(colors)


def ranking(embedding, color, refs, excluded):
    visible = [i for i in range(16) if i not in excluded]
    result = []
    for key, feat, lab in refs:
        visual = np.clip((1 - (feat[visible] * embedding[visible]).sum(axis=1)) / 2, 0, 1)
        col = np.clip(np.linalg.norm(lab[visible] - color[visible], axis=1) / 100, 0, 1)
        result.append((float(np.mean(.8 * visual + .2 * col)), key))
    return [key for _, key in sorted(result)[:5]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_dinov3_model('dinov3_vitb16', pretrained_weight_path=str(WEIGHT)).to(device).eval()
    results = []
    for category in ['splicing_connectors', 'juice_bottle']:
        base = ROOT / 'data/mvtec_loco' / category
        train = sorted((base / 'train/good').glob('*.png'))[:10]
        val = sorted((base / 'validation/good').glob('*.png'))[:2]
        if len(train) != 10 or len(val) != 2:
            raise RuntimeError('need ten training and two validation images')
        refs = []
        for path in train:
            with Image.open(path) as im:
                feat, lab = encode_tiles(im, model, device)
            refs.append((path.name, feat, lab))
        with Image.open(val[0]) as im:
            source = im.convert('RGB')
        with Image.open(val[1]) as im:
            donor = im.convert('RGB')
        if source.size != donor.size:
            raise RuntimeError('source and donor must have the same pixel dimensions')
        original_embedding, original_color = encode_tiles(source, model, device)
        for target in [[0], [5], [10], [15], [0, 1, 4, 5]]:
            altered = source.copy()
            grid = boxes(source.width, source.height)
            for tile_id in target:
                box = grid[tile_id]
                altered.paste(donor.crop(box), box)
            embedding, color = encode_tiles(altered, model, device)
            visible = [i for i in range(16) if i not in target]
            max_diff = float(np.max(np.abs(original_embedding[visible] - embedding[visible])))
            color_diff = float(np.max(np.abs(original_color[visible] - color[visible])))
            a = ranking(original_embedding, original_color, refs, target)
            b = ranking(embedding, color, refs, target)
            results.append(dict(category=category, source=val[0].name, donor=val[1].name, target_tile=target,
                                source_reference_ids=a, replaced_reference_ids=b,
                                visible_embedding_max_abs_diff=max_diff, visible_color_max_abs_diff=color_diff,
                                reference_ids_equal=a == b, target_pixels_changed=any(not np.array_equal(np.asarray(source.crop(grid[i])), np.asarray(donor.crop(grid[i]))) for i in target)))
            print(category, target, results[-1], flush=True)
    with WEIGHT.open('rb') as f:
        weight_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    report = dict(description='Preliminary leakage check, 10 training references per class, one validation pair, four small and one large windows; no component proposals or prediction accuracy.',
                  dinov3_sha256=weight_hash, results=results)
    (OUT / 'tile_isolation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if not all(r['target_pixels_changed'] and r['reference_ids_equal'] and r['visible_embedding_max_abs_diff'] == 0 and r['visible_color_max_abs_diff'] == 0 for r in results):
        raise SystemExit('At least one isolation check failed')


if __name__ == '__main__':
    main()
