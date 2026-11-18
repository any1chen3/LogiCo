"""Full-test object-deficit detection and exact discrete operating-point sPRO."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from utils.spro.util import get_auc_for_max_fpr
from context_prediction.e4_area_local_spro import (
    AREA, BASE, BUDGETS, CATEGORIES, ROOT, DefectsConfig, GroundTruthMap,
    IMAGE_SIZE, labels_at, predict, read_csv,
)
from context_prediction.e4_area_paired_eval import features, scores, split_count

OUT = ROOT / "saved_results/context_e4_area_full_spro"
PREV = ROOT / "saved_results/context_e4_area_paired_eval"
KINDS = ("good", "logical_anomalies", "structural_anomalies")


def write_table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def measure(records, kind, variant):
    group = [r for r in records if r["kind"] in ("good", kind)]
    field = variant + "_score"
    thresholds = sorted({0., *(r[field] for r in group)}, reverse=True)
    den = sum(r["background"] for r in group)
    nr = sum(len(r[variant+"_regions"]) for r in group)
    rates, overlaps = [], []
    for t in thresholds:
        active = [r for r in group if r[field] > t]
        rates.append(sum(r[variant+"_fp"] for r in active) / den)
        overlaps.append(sum(sum(r[variant+"_regions"]) for r in active) / nr)
    # Crossing below zero predicts every pixel, including all previously tied zeros.
    auc = get_auc_for_max_fpr(np.array(rates+[1.]), np.array(overlaps+[1.]), .05)
    points = []
    for budget in BUDGETS:
        idx = max(i for i, rate in enumerate(rates) if rate <= budget)
        points.append(dict(budget=budget, threshold=thresholds[idx], actual_fpr=rates[idx], spro=overlaps[idx]))
        for r in group:
            if r["kind"] != kind:
                continue
            values = r[variant+"_regions"]
            r[f"{variant}_spro_{int(100*budget)}pct"] = float(np.mean(values)) if r[field] > thresholds[idx] else 0.
    count = sum(r["kind"] == kind for r in group)
    return dict(images=count, regions=nr, background_pixels=den, au_spro=float(auc), points=points)


def evaluate(cat, area_models, after, previous):
    dest = OUT / cat
    dest.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data/mvtec_loco" / cat
    seg = ROOT / "data/mvtec_loco_segmentations" / cat
    cfg = DefectsConfig.create_from_list(json.loads((data/"defects_config.json").read_text()))
    setup = json.loads((PREV/"summary.json").read_text())[cat]
    names = setup["labels"]
    with np.load(BASE/cat/"normal_bank.npz") as bank:
        files, occupancy, raw_bank = bank["filenames"].tolist(), bank["spatial_occupancy"], bank["counts"]
    split_bank = np.array([[after[(cat,f,n)] for n in names] for f in files])
    models = [area_models[n] for n in names]
    records = []
    width, height = IMAGE_SIZE[cat]
    for kind in KINDS:
        split = "test/" + kind
        paths = sorted((data/split).glob("*.png"))
        assert paths and {p.name for p in paths} == {p.name for p in (seg/split).glob("*.png")}
        folder = dest / "maps" / kind
        folder.mkdir(parents=True, exist_ok=True)
        for path in paths:
            raw, grid = features(seg/split/path.name, len(names)+1)
            labels = labels_at(seg/split/path.name)
            counted = np.array([split_count(labels==i, model) for i, model in enumerate(models)])
            row = dict(category=cat, kind=kind, image=path.name)
            maps = {}
            for variant, query, library, priors in (("old",raw,raw_bank,[None]*len(names)),("new",counted,split_bank,models)):
                distances, _, _ = scores(query,grid,library,occupancy)
                idx = int(distances.argmin())
                score = float(distances[idx])
                trigger = score > setup[variant+"_threshold"]
                mask = predict(seg,split,path.name,files[idx],query,library[idx],priors) if trigger else np.zeros_like(labels,dtype=np.uint8)
                row.update({variant+"_score":score,variant+"_flag":trigger,variant+"_reference":files[idx]})
                if (cat,split,path.name) in previous:
                    assert np.isclose(score, float(previous[cat,split,path.name][variant+"_score"]), atol=1e-12, rtol=0)
                maps[variant] = mask
            np.savez_compressed(folder/(path.stem+".npz"), **{v:m * row[v+"_score"] for v,m in maps.items()})
            gt = None if kind=="good" else GroundTruthMap.read_from_png_dir(png_dir=str(data/"ground_truth"/kind/path.stem), defects_config=cfg)
            bg = None if gt is None else ~gt.get_or_over_channels()
            row["background"] = width*height if bg is None else int(bg.sum())
            row["defects"] = "" if gt is None else "|".join(ch.defect_config.defect_name for ch in gt.channels)
            for v,m in maps.items():
                full = cv2.resize(m,(width,height),interpolation=cv2.INTER_NEAREST).astype(bool)
                row[v+"_fp"] = int(full.sum() if bg is None else (bg & full).sum())
                row[v+"_regions"] = [] if gt is None else [min(1.,np.count_nonzero(full & ch.bool_array)/ch.get_saturation_area()) for ch in gt.channels]
                for budget in BUDGETS:
                    row[f"{v}_spro_{int(100*budget)}pct"] = None
            records.append(row)
        print(cat,kind,len(paths),flush=True)
    results = {v:{k:measure(records,k,v) for k in KINDS[1:]} for v in ("old","new")}
    for v, entry in results.items():
        entry["mean_au_spro"] = float(np.mean([entry[k]["au_spro"] for k in KINDS[1:]]))
        entry["image_flags"] = {k:sum(r[v+"_flag"] for r in records if r["kind"]==k) for k in KINDS}
    prior_subset=ROOT/"saved_results/context_e4_area_local_spro"/cat/"per_image_spro.csv"
    by_name={r["image"]:r for r in records if r["kind"]=="logical_anomalies"}
    for point in read_csv(prior_subset):
        if point["variant"] in results:
            key = point["variant"]+"_spro_"+str(round(100*float(point["budget"])))+"pct"
            assert abs(by_name[point["image"]][key]-float(point["spro"])) < 1e-10
    write_table(dest/"per_image.csv",records)
    return results


if __name__ == "__main__":
    area=json.loads((AREA/"summary.json").read_text())["models"]
    after={(r["category"],r["image"],r["label"]):int(r["after"]) for r in read_csv(AREA/"per_image_class.csv") if r["split"]=="train/good"}
    previous={(r["category"],r["split"],r["image"]):r for r in read_csv(PREV/"per_image.csv")}
    summary={cat:evaluate(cat,area[cat],after,previous) for cat in CATEGORIES}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    for cat, result in summary.items():
        for v, r in result.items():
            print(cat,v,r["mean_au_spro"],[(k,r[k]["au_spro"],r[k]["points"][0]) for k in KINDS[1:]],r["image_flags"])
