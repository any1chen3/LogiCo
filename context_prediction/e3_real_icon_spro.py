"""E3 retrieval ablation on real logical test defects; score upper sticker interior."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from utils.spro.image import DefectsConfig, GroundTruthMap
from utils.spro.util import get_auc_for_max_fpr
from context_prediction.e3_juice_icon_swap import (
    CACHE, CAT, DATA, ROOT, VISIBLE, WEIGHT, best_ids, digest, encode_tiles,
    inspect, load_dinov3_model, tile_distances,
)

OUT = ROOT / "saved_results/context_e3_real_icon_spro"
MODES = ("fixed", "full", "masked")


def patch(image, box):
    return np.array(image.crop(box).resize((64,64), Image.Resampling.BILINEAR), dtype=np.float32)


def write_csv(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = sorted((DATA/"train/good").glob("*.png"))
    names = np.array([p.stem for p in train])
    features, colors, boxes, templates = [], [], [], {}
    assert digest(WEIGHT) == json.loads((CACHE/"config.json").read_text())["dino_sha256"]
    for p in train:
        with np.load(CACHE/CAT/"train"/(p.stem+".npz")) as z:
            assert str(z["source_sha256"]) == digest(p)
            features.append(z["tile_features"])
            colors.append(z["tile_lab"])
        box,_ = inspect(p)
        boxes.append(box)
        with Image.open(p) as im:
            templates[p.stem] = patch(im.convert("RGB"), box)
    fallback = tuple(np.median(boxes, axis=0).astype(int))
    feats = np.stack(features)
    labs = np.stack(colors)
    order = np.argsort(np.linalg.norm((feats-feats.mean(axis=0)).reshape(len(train),-1),axis=1),kind="stable")
    fixed = tuple(names[order[np.linspace(0,len(train)-1,5,dtype=int)]])
    model = load_dinov3_model("dinov3_vitb16", pretrained_weight_path=str(WEIGHT)).cuda().eval()
    cfg = DefectsConfig.create_from_list(json.loads((DATA/"defects_config.json").read_text()))
    background = {m:np.zeros(256,np.int64) for m in MODES}
    records, regions = [], []
    output_rows = []
    for kind in ("good", "logical_anomalies"):
        for path in sorted((DATA/"test"/kind).glob("*.png")):
            with Image.open(path) as im:
                img = im.convert("RGB")
            w,h=img.size
            f,l=encode_tiles(img,model,"cuda:0")
            dist=tile_distances(dict(tile_features=f,tile_lab=l),feats,labs)
            ranks=dict(fixed=fixed,full=best_ids(dist,names,list(range(16))),masked=best_ids(dist,names,VISIBLE))
            # Read the observed sticker only after ranking.
            reason = ""
            try:
                box,_=inspect(path)
                x,y,X,Y=box
                if X<=x or Y<=y: raise ValueError("empty observed sticker")
            except ValueError as exc:
                box=fallback
                reason=str(exc)
            # Keep the target inside the fixed hidden tile window for all methods.
            x,y,X,Y=box
            box=(max(x,w//4),max(y,h//4),min(X,3*w//4),min(Y,3*h//4))
            if box[2]<=box[0] or box[3]<=box[1]:
                box=fallback
                reason="outside central window"
            x,y,X,Y=box
            sample=patch(img,box)
            gt=None if kind=="good" else GroundTruthMap.read_from_png_dir(png_dir=str(DATA/"ground_truth"/kind/path.stem),defects_config=cfg)
            bg=None if gt is None else ~gt.get_or_over_channels()
            for m,ids in ranks.items():
                # Absolute RGB residual per aligned pixel, median over the 5 references.
                residual=np.median([np.abs(sample-templates[k]).mean(axis=2) for k in ids],axis=0)
                heat=np.zeros((h,w),np.uint8)
                heat[y:Y,x:X]=np.rint(cv2.resize(residual,(X-x,Y-y),interpolation=cv2.INTER_LINEAR)).astype(np.uint8)
                background[m]+=np.bincount(heat.ravel() if bg is None else heat[bg],minlength=256)
                records.append(dict(split=kind,image=path.name,mode=m,refs=";".join(ids),fallback=reason,box=str(box)))
                if gt:
                    for j,ch in enumerate(gt.channels):
                        area=ch.get_saturation_area()
                        hist=np.bincount(heat[ch.bool_array],minlength=256)
                        overlap=np.minimum(1.,(hist.sum()-hist.cumsum())/area)
                        bound=min(1.,ch.bool_array[y:Y,x:X].sum()/area)
                        regions.append((m,path.name,j,ch.defect_config.defect_name,overlap,bound))
        print("finished",kind,flush=True)
    write_csv(OUT/"references.csv",records)
    summaries=[]
    for mode in MODES:
        hist=background[mode]
        rates=(hist.sum()-hist.cumsum())/hist.sum()
        group=[r for r in regions if r[0]==mode]
        for label in ["ALL"]+sorted({r[3] for r in group}):
            items=group if label=="ALL" else [r for r in group if r[3]==label]
            curve=np.mean([r[4] for r in items],axis=0)
            auc=get_auc_for_max_fpr(np.r_[rates[::-1],1.],np.r_[curve[::-1],1.],.05)
            thresholds=[int(np.flatnonzero(rates<=b)[0]) for b in (.01,.03,.05)]
            entry=dict(mode=mode,defect=label,images=len({r[1] for r in items}),regions=len(items),au_spro=float(auc),max_roi_spro=float(np.mean([r[5] for r in items])))
            for budget,t in zip((1,3,5),thresholds):
                entry.update({f"threshold_{budget}":t,f"actual_fpr_{budget}":rates[t],f"spro_{budget}":curve[t]})
            summaries.append(entry)
            if label!="ALL":
                for _,name,j,defect,spro,bound in items:
                    output_rows.append(dict(mode=mode,image=name,channel=j,defect=defect,spro_1=spro[thresholds[0]],spro_3=spro[thresholds[1]],spro_5=spro[thresholds[2]],max_roi_spro=bound))
    write_csv(OUT/"summary.csv",summaries)
    write_csv(OUT/"regions.csv",output_rows)
    assert all(sum(background[m])==sum(background["fixed"]) for m in MODES)
    print("complete",len(records)//3,"images",flush=True)


if __name__=="__main__":
    main()
