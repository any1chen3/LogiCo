"""Independently verify the 1% work points using direct binary masks."""
import csv,json
from pathlib import Path
import numpy as np
from utils.spro.image import DefectsConfig
from breakfast_ab_diagnosis import ROOT, DATA, OUT, DEST, gt_for, maps_for

with (OUT/'scores.csv').open(encoding='utf-8',newline='') as f: rows=[r for r in csv.DictReader(f) if r['defect_type'] in ('good','logical_anomalies')]
cfg=DefectsConfig.create_from_list(json.loads((DATA/'defects_config.json').read_text()))
result=json.loads((DEST/'diagnosis.json').read_text(encoding='utf-8'))
byimg={}
fp={'A':0,'B':0}; den=0; cover={'A':[],'B':[]}
for r in rows:
    gt=gt_for(r,cfg); maps=maps_for(r); bg=None if gt is None else ~gt.get_or_over_channels(); den+=maps['A'].size if bg is None else int(bg.sum())
    for v in ('A','B'):
        binary=maps[v].astype(np.float64)>result['variants'][v]['points']['0.01']['threshold']
        fp[v]+=int(binary.sum()) if bg is None else int(np.count_nonzero(binary & bg))
        if gt is not None:
            for channel in gt.channels:cover[v].append(min(np.count_nonzero(binary & channel.bool_array)/channel.get_saturation_area(),1.0))
for v in ('A','B'):
    actual=fp[v]/den; s=float(np.mean(cover[v])); expected=result['variants'][v]['points']['0.01']
    print(v,'direct FPR=',actual,'sPRO=',s,'expected=',expected['actual_fpr'],expected['spro'],'regions=',len(cover[v]),flush=True)
    assert abs(actual-expected['actual_fpr'])<1e-12 and abs(s-expected['spro'])<1e-12
print('Direct work-point check passed')
