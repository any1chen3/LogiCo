"""Paired operating-point sPRO for unmatched reference instances, no GT in prediction."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from context_prediction.e4_instance_reference_area import OUT as AREA, components, gated, labels_at
from context_prediction.e4_shared_normal_configuration import ROOT, OUT as BASE, CATEGORIES
from utils.spro.image import DefectsConfig, GroundTruthMap
from utils.spro.spro import IMAGE_SIZE

OUT = ROOT / "saved_results/context_e4_area_local_spro"
BUDGETS = (.01, .03, .05)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def instances(labels, cls, model):
    masks = []
    for part, size in components(labels == cls):
        if model and model["training_multi_region_images"] >= 10:
            median = model["median_area"]
            if size >= 1.5 * median:
                _, n, _, pieces = gated(part)
                sizes = np.bincount(pieces.ravel(), minlength=n + 1)[1:]
                if n > 1 and np.all((sizes >= .5 * median) & (sizes <= 1.8 * median)):
                    masks.extend(pieces == j for j in range(1, n + 1))
                    continue
        masks.append(part)
    return masks


def predict(seg, split, name, ref, observed, expected, models):
    query = labels_at(seg / split / name)
    reference = labels_at(seg / "train/good" / ref)
    out = np.zeros(query.shape, np.uint8)
    for cls, (a, b) in enumerate(zip(observed, expected)):
        if a >= b:
            continue
        q = instances(query, cls, models[cls])
        r = instances(reference, cls, models[cls])
        assert (len(q), len(r)) == (a, b)
        matched = set()
        if q:
            qxy = np.array([np.argwhere(mask).mean(axis=0) for mask in q])
            rxy = np.array([np.argwhere(mask).mean(axis=0) for mask in r])
            distance = np.linalg.norm(qxy[:, None] - rxy[None], axis=2)
            matched.update(linear_sum_assignment(distance)[1])
        for j, mask in enumerate(r):
            if j not in matched:
                out[mask & (query != cls)] = 1
    return out


def evaluate(cat, rows, models):
    folder = OUT / cat
    folder.mkdir(parents=True, exist_ok=True)
    ab = ROOT / "saved_results" / ("breakfast_ab" if cat == "breakfast_box" else "ab_splicing_connectors")
    previous = {(r["split"], r["image"]): r for r in read_csv(BASE / cat / "per_image.csv")}
    prior = json.loads((ab / "diagnosis/diagnosis.json").read_text(encoding="utf-8"))
    data = ROOT / "data/mvtec_loco" / cat
    seg = ROOT / "data/mvtec_loco_segmentations" / cat
    cfg = DefectsConfig.create_from_list(json.loads((data/"defects_config.json").read_text()))
    size = IMAGE_SIZE[cat]
    den = 0
    predictions = {"old": [], "new": []}
    metrics = []
    for row in rows:
        split, name = row["split"], row["image"]
        if row["category"] != cat or not split.startswith("test/"):
            continue
        target = row["target_subset"] == "True"
        baseline = previous[split, name]
        masks = {}
        for variant in predictions:
            mask = np.zeros((256,256), np.uint8)
            if row[variant + "_flag"] == "True":
                if variant == "new":
                    ref = row["reference"]
                    observed, expected = json.loads(row["observed"]), json.loads(row["expected"])
                    priors = models
                else:
                    ref = baseline["nearest_training_image"]
                    observed, expected = [list(map(int, baseline[k].split(";"))) for k in ("observed_counts","reference_counts")]
                    priors = [None] * len(models)
                mask = predict(seg, split, name, ref, observed, expected, priors)
            masks[variant] = mask
        if target:
            np.savez_compressed(folder/(Path(name).stem+".npz"), **{v:m * float(row[v+"_score"]) for v,m in masks.items()})
        gt = None if split == "test/good" else GroundTruthMap.read_from_png_dir(
            png_dir=str(data/"ground_truth"/"logical_anomalies"/Path(name).stem), defects_config=cfg)
        bg = None if gt is None else ~gt.get_or_over_channels()
        den += size[0]*size[1] if bg is None else int(bg.sum())
        for v, mask in masks.items():
            full = cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST).astype(bool)
            fp = int(full.sum() if bg is None else np.count_nonzero(full & bg))
            channels = [] if gt is None else [min(1.,np.count_nonzero(full & ch.bool_array)/ch.get_saturation_area()) for ch in gt.channels]
            predictions[v].append((name,target,float(row[v+"_score"]),fp,channels))
        if target:
            with np.load(ab/"maps"/"logical_anomalies"/(Path(name).stem+".npz")) as z:
                for v in ("A","B"):
                    full = cv2.resize(z[v],size,interpolation=cv2.INTER_LINEAR).astype(np.float64)
                    for budget in BUDGETS:
                        pt = prior["variants"][v]["points"][str(budget)]
                        binary = full > pt["threshold"]
                        spro = np.mean([min(1., np.count_nonzero(binary & ch.bool_array)/ch.get_saturation_area()) for ch in gt.channels])
                        metrics.append((v,budget,pt["threshold"],pt["actual_fpr"],name,spro))
    assert den == prior["protocol"]["fpr_denominator"]
    for v, samples in predictions.items():
        for budget in BUDGETS:
            for cutoff in sorted({0., *(s[2] for s in samples)}):
                fp = sum(s[3] for s in samples if s[2] > cutoff)
                if fp/den <= budget:
                    break
            for name,target,score,_,channels in samples:
                if target:
                    metrics.append((v,budget,cutoff,fp/den,name,np.mean(channels) if score > cutoff else 0.))
    with (folder/"per_image_spro.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["variant","budget","threshold","actual_fpr","image","spro"])
        w.writerows(metrics)
    summary = []
    for v in ("A","B","old","new"):
        for budget in BUDGETS:
            selected = [r for r in metrics if r[:2] == (v,budget)]
            summary.append(dict(variant=v,budget=budget,threshold=selected[0][2],actual_fpr=selected[0][3],
                                n=len(selected),spro=float(np.mean([r[-1] for r in selected]))))
    return summary


if __name__ == "__main__":
    inputs = read_csv(ROOT/"saved_results/context_e4_area_paired_eval/per_image.csv")
    area = json.loads((AREA/"summary.json").read_text(encoding="utf-8"))["models"]
    results = {}
    for category in CATEGORIES:
        names = json.loads((BASE/category/"summary.json").read_text())["foreground_labels"]
        results[category] = evaluate(category, inputs, [area[category][n] for n in names])
        print(category, results[category], flush=True)
    (OUT/"summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
