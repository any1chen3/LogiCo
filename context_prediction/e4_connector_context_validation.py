"""Extend the fixed-band side-context retrieval probe on normal validation images."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from context_prediction.e4_full_line_mask_probe import (
    ROOT, DATA, SEG, OUT as PREVIOUS, CACHE, WEIGHT, digest, config_hash,
    signature, visible_strips, rank, read, tile_distances, best_ids, load_dinov3_model,
)

OUT = ROOT / 'saved_results/context_e4_connector_context_validation'
BAND = (55, 207)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = sorted((DATA / 'train/good').glob('*.png'))
    with np.load(PREVIOUS / 'train_side_strips.npz') as z:
        names, hashes = z['names'], z['hashes']
        features, labs = z['features'], z['labs']
        if list(z['band']) != list(BAND) or str(z['weight_hash']) != digest(WEIGHT):
            raise RuntimeError('Side strip bank has a different band or backbone')
        if list(names) != [p.stem for p in train] or list(hashes) != [digest(p) for p in train]:
            raise RuntimeError('Side strip bank does not match reference images')
    paths = sorted((CACHE / 'splicing_connectors/train').glob('*.npz'))
    if len(paths) != len(train) or config_hash(CACHE / 'config.json') != digest(WEIGHT):
        raise RuntimeError('Full image cache does not match encoder or bank size')
    cached = [(p, read(p)) for p in paths]
    full_features = np.stack([d['tile_features'] for _, d in cached])
    full_labs = np.stack([d['tile_lab'] for _, d in cached])
    full_names = np.asarray([p.stem for p, _ in cached])
    if list(full_names) != list(names):
        raise RuntimeError('Reference name order mismatch')
    mean = full_features.mean(axis=0)
    ordered = sorted((float(np.linalg.norm(d['tile_features'] - mean)), p.stem) for p, d in cached)
    fixed = [ordered[i][1] for i in np.linspace(0, len(ordered)-1, 5, dtype=int)]
    signatures = {name: signature(SEG / 'train/good' / (name + '.png'))[0] for name in names}
    model = load_dinov3_model('dinov3_vitb16', pretrained_weight_path=str(WEIGHT)).cuda().eval()
    images = [DATA / 'validation/good' / f'{i:03d}.png' for i in range(5, 30)]
    rows, checks = [], []
    for idx, path in enumerate(images):
        with Image.open(path) as image: image = image.convert('RGB')
        with Image.open(images[(idx+1) % len(images)]) as donor: donor = donor.convert('RGB')
        if image.size != donor.size: raise ValueError('Incompatible original dimensions')
        f, lab = visible_strips(image, *BAND, model)
        side_ids = rank(f, lab, features, labs, names)
        x1, x2 = image.width * BAND[0] // 256, image.width * BAND[1] // 256
        altered = image.copy(); altered.paste(donor.crop((x1, 0, x2, image.height)), (x1, 0))
        f2, lab2 = visible_strips(altered, *BAND, model)
        side_ids_2 = rank(f2, lab2, features, labs, names)
        q = read(CACHE / 'splicing_connectors/validation' / (path.stem+'.npz'))
        if str(q['source_sha256']) != digest(path): raise RuntimeError('Validation cache source changed')
        full_ids = best_ids(tile_distances(q, full_features, full_labs), full_names, list(range(16)))
        changed = bool(np.any(np.asarray(image.crop((x1,0,x2,image.height))) != np.asarray(donor.crop((x1,0,x2,image.height)))))
        checks.append(dict(image=path.stem, donor=images[(idx+1)%len(images)].stem, target_changed=changed,
            max_feature_difference=float(np.max(np.abs(f-f2))), max_lab_difference=float(np.max(np.abs(lab-lab2))),
            same_reference_ids=bool(side_ids == side_ids_2)))
        obs, reason = signature(SEG / 'validation/good' / path.name)
        if obs is None:
            for method, ids in [('fixed',fixed),('full',full_ids),('side_strips',side_ids)]:
                rows.append(dict(image=path.stem,method=method,reference_ids=';'.join(ids),status=reason,
                    line_pixels_outside='',usable=0,same_class=0,mean_endpoint_all='',mean_endpoint_same_class=''))
            continue
        line=obs['line']['mask']; outside=line.copy();outside[:,BAND[0]:BAND[1]]=False
        for method, ids in [('fixed',fixed),('full',full_ids),('side_strips',side_ids)]:
            usable=[signatures[x] for x in ids if signatures[x] is not None]
            same=[v for v in usable if v['line']['class_id']==obs['line']['class_id']]
            def avg(items):
                values=[float(np.mean(np.abs(np.asarray(v['heights'])-obs['heights']))) for v in items]
                return float(np.mean(values)) if values else ''
            rows.append(dict(image=path.stem,method=method,reference_ids=';'.join(ids),status='ok',
                line_pixels_outside=int(outside.sum()),usable=len(usable),same_class=len(same),
                mean_endpoint_all=avg(usable),mean_endpoint_same_class=avg(same)))
        print('checked',path.stem,flush=True)
    for name,items in [('per_image.csv',rows),('replacement_checks.csv',checks)]:
        with (OUT / name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=items[0]);writer.writeheader();writer.writerows(items)
    result={}
    for method in ('fixed','full','side_strips'):
        group=[r for r in rows if r['method']==method and r['status']=='ok']
        hidden=[r for r in group if r['line_pixels_outside']==0]
        def agg(rs):
            values=[r['mean_endpoint_all'] for r in rs if r['mean_endpoint_all']!='']
            return dict(images=len(rs),mean_same_class=float(np.mean([r['same_class'] for r in rs])) if rs else None,
                        endpoint_images=len(values),mean_endpoint_error=float(np.mean(values)) if values else None)
        result[method]=dict(all_valid=agg(group),auto_line_completely_hidden=agg(hidden))
    summary=dict(protocol='Exploratory normal validation 005-029; fixed band 55:207 designed on 000-004. Evaluation 010-019 previously seen in thumbnail overview, 010-029 were used in window audits. Side crop features are computed from outside the band on original RGB; line segmentation used only after retrieval for diagnostic scoring.',
        warning='Full-image retrieval includes target line and is an upper/reference comparator, not leakage-safe. Automatic segmentation and endpoint proxy are not human GT. This does not assess abnormal missing wires or sPRO.',
        bands_256=list(BAND), fixed_reference_ids=fixed, evaluation_count=len(images),
        target_replacement=dict(changed=sum(c['target_changed'] for c in checks),same_ids=sum(c['same_reference_ids'] for c in checks),
            max_feature_difference=max(c['max_feature_difference'] for c in checks),max_lab_difference=max(c['max_lab_difference'] for c in checks)),results=result)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__': main()
