"""Compare a fresh E0 export with the historical image-keyed baseline."""
import csv
import json
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'saved_results/ab_splicing_connectors'
NEW = ROOT / 'saved_results/context_e0/best'


def rows_at(directory):
    with (directory / 'scores.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    result = {r['image_path']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate image paths')
    return result


def main():
    old, new = rows_at(OLD), rows_at(NEW)
    if old.keys() != new.keys():
        raise ValueError('historical and new image paths differ')
    original = json.loads((OLD / 'inference.json').read_text())
    current = json.loads((NEW / 'inference.json').read_text())
    if original['checkpoint_sha256'] != current['checkpoint_sha256']:
        raise ValueError('checkpoint differs')
    cfg = yaml.safe_load((ROOT / 'configs/train/mvtec_loco.yaml').read_text())
    k = int(cfg['resize_mask'] ** 2 * cfg['max_ratio'])
    differences = {name: 0. for name in ('CRN', 'SRN', 'SMD', 'A', 'B')}
    max_score_diff, max_readback_diff = 0., 0.
    shapes = {}
    for key in sorted(old):
        o, n = old[key], new[key]
        if (o['defect_type'], o['label']) != (n['defect_type'], n['label']):
            raise ValueError('image labels differ: ' + key)
        max_score_diff = max(max_score_diff, abs(float(o['score_A']) - float(n['score_A'])))
        with np.load(o['map_path']) as before, np.load(n['map_path']) as after:
            for name in differences:
                if before[name].shape != after[name].shape or not np.isfinite(after[name]).all():
                    raise ValueError('invalid map: ' + key + ' ' + name)
                shapes[name] = list(after[name].shape)
                differences[name] = max(differences[name], float(np.max(np.abs(before[name] - after[name]))))
            flat = after['A'].ravel()
            score = float(np.sort(flat)[-k:].mean())
            max_readback_diff = max(max_readback_diff, abs(score - float(n['score_A'])))
    result = dict(count=len(new), checkpoint_sha256=current['checkpoint_sha256'], shapes=shapes,
                  max_map_abs_difference=differences, max_score_difference=max_score_diff,
                  max_score_readback_difference=max_readback_diff, tolerance=1e-5,
                  passed=max(list(differences.values()) + [max_score_diff, max_readback_diff]) <= 1e-5)
    (NEW / 'alignment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
