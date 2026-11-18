"""Full-resolution 1% FPR operating points on a fixed 0.001-score threshold grid."""
import json
import csv
from pathlib import Path

import cv2
import numpy as np
from context_prediction.e4_ab_area_fusion import CATEGORIES, OUT, ROOT, ab_folder
from utils.spro.image import GroundTruthMap, DefectsConfig
from utils.spro.spro import IMAGE_SIZE

STEP = .001


def run(cat):
    dest = OUT / cat
    alpha = np.float32(json.loads((dest/"calibration.json").read_text())["alpha"])
    source = ab_folder(cat)
    with (source/"scores.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    target = set(json.loads((ROOT/"saved_results/context_e4_shared_normal_configuration"/cat/"summary.json").read_text())["target_ids"])
    cfg_path = ROOT/"data/mvtec_loco"/cat
    cfg = DefectsConfig.create_from_list(json.loads((cfg_path/"defects_config.json").read_text()))
    variants = ("A","A_plus_E","B","B_plus_E")
    kinds = ("good","logical_anomalies","structural_anomalies")
    # The central offset permits negative heat scores. Check range on every image.
    low, bins = -64., 262144
    hist = {(v,k):np.zeros(bins,dtype=np.int64) for v in variants for k in kinds}
    overlap = {(v,k):np.zeros(bins) for v in variants for k in (*kinds,"target")}
    channels = dict.fromkeys((*kinds,"target"),0)
    for row in rows:
        k = row["defect_type"]
        name = Path(row["image_path"]).stem
        gt = None if k=="good" else GroundTruthMap.read_from_png_dir(png_dir=str(cfg_path/"ground_truth"/k/name),defects_config=cfg)
        bg = None if gt is None else ~gt.get_or_over_channels()
        subset = k=="logical_anomalies" and name+".png" in target
        if gt:
            channels[k]+=len(gt.channels)
            if subset:
                channels["target"]+=len(gt.channels)
        with np.load(ROOT/"saved_results/context_e4_area_full_spro"/cat/"maps"/k/(name+".npz")) as z:
            inc = alpha*z["new"].astype(np.float32)
        with np.load(row["map_path"]) as z:
            for base in ("A","B"):
                for fused in (False,True):
                    v = base+"_plus_E" if fused else base
                    heat = z[base]+inc if fused else z[base]
                    full = cv2.resize(heat,IMAGE_SIZE[cat],interpolation=cv2.INTER_LINEAR)
                    ids = np.floor((full.astype(np.float64)-low)/STEP).astype(np.int32)
                    assert ids.min()>=0 and ids.max()<bins
                    hist[v,k]+=np.bincount(ids.ravel() if bg is None else ids[bg],minlength=bins)
                    if gt:
                        for ch in gt.channels:
                            hits=np.bincount(ids[ch.bool_array],minlength=bins)[::-1].cumsum()[::-1]
                            spro=np.minimum(hits/ch.get_saturation_area(),1.)
                            overlap[v,k]+=spro
                            if subset:
                                overlap[v,"target"]+=spro
    results=[]
    for v in variants:
        for k in kinds[1:]:
            bg = hist[v,k]+hist[v,"good"]
            rates = bg[::-1].cumsum()[::-1]/bg.sum()
            i = int(np.flatnonzero(rates<=.01)[0])
            results.append(dict(category=cat,variant=v,group=k,threshold=low+i*STEP,actual_fpr=rates[i],spro=overlap[v,k][i]/channels[k],
                target_spro=overlap[v,"target"][i]/channels["target"] if k=="logical_anomalies" else None))
    return results


if __name__=="__main__":
    rows=[]
    for cat in CATEGORIES:
        rows.extend(run(cat))
        print(cat,"complete",flush=True)
    with (OUT/"operating_points.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
