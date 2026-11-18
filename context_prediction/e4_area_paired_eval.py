"""Paired image-level evaluation with frozen normal-reference area splitting."""
import csv
import json

import numpy as np

from context_prediction.e4_instance_reference_area import OUT as AREA, components, gated, labels_at
from context_prediction.e4_shared_normal_configuration import CATEGORIES, OUT as BASE, ROOT, features, scores

OUT = ROOT / "saved_results/context_e4_area_paired_eval"


def split_count(mask, model):
    parts = components(mask)
    count = len(parts)
    median = model["median_area"]
    if median is None or model["training_multi_region_images"] < 10:
        return count
    for part, size in parts:
        if size < 1.5 * median:
            continue
        _, number, _, pieces = gated(part)
        sizes = np.bincount(pieces.ravel(), minlength=number + 1)[1:]
        if number > 1 and np.all((sizes >= 0.5 * median) & (sizes <= 1.8 * median)):
            count += number - 1
    return count


def evaluate(category, models, cached_counts):
    folder = BASE / category
    old = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    names = old["foreground_labels"]
    with np.load(folder / "normal_bank.npz") as bank:
        files = bank["filenames"].tolist()
        occupancy = bank["spatial_occupancy"]
        baseline = bank["counts"]
    counts = np.array([[cached_counts[(category, "train/good", f, n)] for n in names] for f in files])
    calibration = []
    for i, (count, grid) in enumerate(zip(counts, occupancy)):
        distance, _, _ = scores(count, grid, counts, occupancy)
        distance[i] = np.inf
        calibration.append(distance.min())
    threshold = float(np.quantile(calibration, .99, method="higher"))
    with (folder / "per_image.csv").open(encoding="utf-8-sig") as stream:
        prior_rows = list(csv.DictReader(stream))
    results = []
    for row in prior_rows:
        split, filename = row["split"], row["image"]
        path = ROOT / "data/mvtec_loco_segmentations" / category / split / filename
        raw, grid = features(path, len(names) + 1)
        previous_score = float(scores(raw, grid, baseline, occupancy)[0].min())
        if not np.isclose(previous_score, float(row["score"]), atol=1e-10, rtol=0):
            raise ValueError(f"Baseline mismatch for {path}")
        labels = labels_at(path)
        query = np.array([split_count(labels == i, models[n]) for i, n in enumerate(names)])
        if split == "validation/good":
            assert query.tolist() == [cached_counts[(category, split, filename, n)] for n in names]
        distance, count_difference, spatial_difference = scores(query, grid, counts, occupancy)
        index = int(distance.argmin())
        results.append(dict(category=category, split=split, image=filename, defect=row["defect"],
                            target_subset=split == "test/logical_anomalies" and filename in old["target_ids"],
                            old_score=previous_score, new_score=float(distance[index]),
                            old_flag=row["flagged"] == "True", new_flag=bool(distance[index] > threshold),
                            observed=query.tolist(), expected=counts[index].tolist(), reference=files[index],
                            count_deficit=float(count_difference[index]), spatial_deficit=float(spatial_difference[index])))
    groups = {}
    for key in ("validation/good", "test/good", "test/logical_anomalies", "target_subset"):
        rows = [r for r in results if (r["target_subset"] if key == "target_subset" else r["split"] == key)]
        groups[key] = dict(n=len(rows), before=sum(r["old_flag"] for r in rows), after=sum(r["new_flag"] for r in rows),
                           gained=[r["image"] for r in rows if r["new_flag"] and not r["old_flag"]],
                           lost=[r["image"] for r in rows if r["old_flag"] and not r["new_flag"]])
    return results, dict(old_threshold=old["loo_threshold"], new_threshold=threshold, labels=names, groups=groups)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = json.loads((AREA / "summary.json").read_text(encoding="utf-8"))["models"]
    with (AREA / "per_image_class.csv").open(encoding="utf-8-sig") as stream:
        counts = {(r["category"], r["split"], r["image"], r["label"]): int(r["after"]) for r in csv.DictReader(stream)}
    all_rows, summary = [], {}
    for category in CATEGORIES:
        rows, summary[category] = evaluate(category, models[category], counts)
        all_rows.extend(rows)
    with (OUT / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0])
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for row in all_rows:
        if row["category"] == "breakfast_box" and row["target_subset"] and row["image"] in ("062.png", "069.png"):
            print(row)


if __name__ == "__main__":
    main()
