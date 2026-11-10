"""Summarize map differences inside and outside labeled defect regions."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'saved_results/breakfast_ab'
DATA = ROOT / 'data/mvtec_loco/breakfast_box'
configs = {x['pixel_value']: x['defect_name'] for x in json.loads((DATA/'defects_config.json').read_text())}

def main():
    with (OUT/'scores.csv').open(encoding='utf-8', newline='') as f: rows = list(csv.DictReader(f))
    results=[]
    for row in rows:
        if row['defect_type'] != 'logical_anomalies': continue
        folder=DATA/'ground_truth'/row['defect_type']/Path(row['image_path']).stem
        with np.load(row['map_path']) as maps:
            a=maps['A'];b=maps['B'];delta=a-b
        for path in sorted(folder.glob('*.png')):
            with Image.open(path) as img:
                full=np.asarray(img)
                ids=np.unique(full);ids=ids[ids>0]
                mask=np.asarray(img.resize((a.shape[1],a.shape[0]),Image.Resampling.NEAREST))>0
            if not mask.any() or not len(ids): continue
            results.append(dict(image=Path(row['image_path']).name,annotation=path.name,defect_type=configs.get(int(ids[0]),str(ids[0])),pixels=int(mask.sum()),a_mean_inside=float(a[mask].mean()),b_mean_inside=float(b[mask].mean()),delta_inside=float(delta[mask].mean()),delta_outside=float(delta[~mask].mean()) if (~mask).any() else None))
    with (OUT/'evidence_by_annotation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
    print(f'Saved {len(results)} annotated regions')

if __name__=='__main__': main()
