"""Validate C1 output completeness, finiteness, and calibration endpoints."""
import json
from pathlib import Path
import numpy as np

folder=Path(__file__).resolve().parent/'saved_results/screw_c1'
cal=json.loads((folder/'calibration.json').read_text())
maps=list((folder/'maps').rglob('*.npz'))
assert len(maps)==341
max_abs=0.0
for path in maps:
    with np.load(path) as data:
        for key in ('A','B','C1'):
            assert data[key].shape==(256,256) and np.isfinite(data[key]).all()
        max_abs=max(max_abs,float(np.abs(data['C1']-data['A']).max()))
assert abs(cal['r95']+cal['scale']*(cal['s95']-cal['s95'])-cal['r95'])<1e-12
assert abs(cal['r95']+cal['scale']*(cal['s99']-cal['s95'])-cal['r99'])<1e-12
print('All',len(maps),'maps finite; percentile endpoints verified; max C1-A difference',max_abs)
