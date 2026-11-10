"""End-to-end compare accelerated sPRO against the original saved result."""
import csv,json,os
from pathlib import Path
import numpy as np
from utils.spro.image import DefectsConfig
from utils.spro.spro import read_maps,get_auc_spro_results
from utils.spro.aggregation import MetricsAggregator
from ab_spro_acceleration import validate_and_install,clear
ROOT=Path(__file__).resolve().parent
category=os.environ.get('LOGICO_AB_CATEGORY','juice_bottle')
out=ROOT/'saved_results'/('breakfast_ab' if category=='breakfast_box' else 'ab_'+category)
with (out/'scores.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
data=ROOT/'data/mvtec_loco'/category
cfg=DefectsConfig.create_from_list(json.loads((data/'defects_config.json').read_text()))
maps=[]
for row in rows:
    with np.load(row['map_path']) as f:maps.append(f['A'])
gt,pred=read_maps([r['image_path'] for r in rows],maps,category,cfg)
validate_and_install(gt,pred)
np.random.seed(1)
m=MetricsAggregator(gt_maps=gt,anomaly_maps=pred,parallel_workers=None,parallel_niceness=20).run(curve_max_distance=.001,max_steps=10)
actual=get_auc_spro_results(m)['auc_spro']
expected=json.loads((out/'spro_results.json').read_text())['variants']['A']
for kind,key in [('logical_anomalies','logical_au_spro'),('structural_anomalies','structural_au_spro')]:
    value=float(actual[kind][.05]);print(kind,value,'original',expected[key],flush=True)
    assert abs(value-expected[key]) < 1e-10, 'Accelerated result differs from original'
clear()
print('Full-dataset original vs accelerated sPRO matched',flush=True)
