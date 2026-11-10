"""Compute local LOCO sPRO for paired A/B maps and specified defect types."""
import csv
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np
from utils.spro.image import DefectsConfig
from utils.spro.spro import read_maps, get_auc_spro_results, get_auc_spros_for_metrics
from utils.spro.aggregation import MetricsAggregator
from ab_spro_acceleration import validate_and_install, clear

ROOT = Path(__file__).resolve().parent
CATEGORY = os.environ.get('LOGICO_AB_CATEGORY', 'breakfast_box')
OUT = Path(os.environ['LOGICO_AB_OUT']) if os.environ.get('LOGICO_AB_OUT') else ROOT / 'saved_results' / ('breakfast_ab' if CATEGORY == 'breakfast_box' else 'ab_' + CATEGORY)
DATA = ROOT / 'data/mvtec_loco' / CATEGORY
GROUPS = {
    'missing_components': ['missing_almonds', 'missing_bananas', 'missing_toppings', 'missing_cereals', 'missing_cereals_and_toppings'],
    'extra_fruit_4_tangerines': ['0_nectarines_4_tangerines'],
    'overflow_separate': ['overflow'],
}
if CATEGORY != 'breakfast_box':
    configs = json.loads((DATA / 'defects_config.json').read_text())
    names = [x['defect_name'] for x in configs]
    GROUPS = {
        'missing_explicit': [n for n in names if 'missing' in n],
        'extra_explicit': [n for n in names if 'additional' in n or n.startswith('extra_')],
        'misplacement_explicit': [n for n in names if any(x in n for x in ('swapped', 'misplaced', 'location', 'not_plugged', 'flipped', 'rotated'))],
    }
    GROUPS = {k:v for k,v in GROUPS.items() if v}

def main():
    with (OUT / 'scores.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    cfg = DefectsConfig.create_from_list(json.loads((DATA / 'defects_config.json').read_text()))
    paths = [r['image_path'] for r in rows]
    output = {'protocol': {'metric':'AU-sPRO at max FPR=0.05', 'implementation':'local utils/spro, original full-image refinement max_steps=10 curve_max_distance=0.001', 'threshold_sampling_seed':1, 'subset_protocol':'defect-name filter within logical anomalies + normal images; shared all-logical FPR curve', 'groups':GROUPS}, 'variants':{}}
    existing = OUT / 'spro_results.json'
    if existing.exists():
        prior = json.loads(existing.read_text(encoding='utf-8'))
        if prior.get('protocol') == output['protocol']:
            output['variants'] = prior.get('variants', {})
    for variant in os.environ.get('LOGICO_AB_VARIANTS', 'A,B').split(','):
        if variant in output['variants']:
            print(f'Reusing completed {variant} sPRO results', flush=True)
            continue
        print(f'Loading {variant} maps', flush=True)
        maps = []
        for row in rows:
            with np.load(row['map_path']) as file:
                maps.append(file[variant])
        gt, predictions = read_maps(paths, maps, CATEGORY, cfg)
        counts = Counter()
        image_counts = Counter()
        for row, g in zip(rows, gt):
            if row['defect_type'] != 'logical_anomalies' or g is None: continue
            names = [c.defect_config.defect_name for c in g.channels]
            counts.update(names)
            for group, choices in GROUPS.items():
                if any(x in choices for x in names): image_counts[group] += 1
        print(f'Computing {variant} local sPRO', flush=True)
        validate_and_install(gt, predictions)
        print('Exact counting acceleration validated against original on normal and anomalous samples', flush=True)
        np.random.seed(1)
        m = MetricsAggregator(gt_maps=gt, anomaly_maps=predictions, parallel_workers=None, parallel_niceness=20).run(curve_max_distance=0.001, max_steps=10)
        overall = get_auc_spro_results(m)['auc_spro']
        log_images = [a for a in m.anomaly_maps if Path(a.file_path).parent.name in ('logical_anomalies', 'good')]
        logical = m.reduce_to_images(log_images)
        subsets = {}
        for name, names in GROUPS.items():
            number = sum(counts[n] for n in names)
            value = get_auc_spros_for_metrics(logical, filter_defect_names_for_spro=names)[0.05] if number else None
            subsets[name] = {'au_spro': float(value) if value is not None else None, 'images':image_counts[name], 'defect_channels':number, 'by_defect':{n:counts[n] for n in names}}
        output['variants'][variant] = {'logical_au_spro':float(overall['logical_anomalies'][0.05]), 'structural_au_spro':float(overall['structural_anomalies'][0.05]), 'mean_au_spro':float(overall['mean'][0.05]), 'subsets':subsets}
        (OUT / 'spro_results.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(output['variants'][variant], ensure_ascii=False, indent=2), flush=True)
        clear()
        del gt, predictions, maps, m, logical
    print('Complete', flush=True)

if __name__ == '__main__': main()
