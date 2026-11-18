"""Validation calibration and paired A/B fusion with frozen area-deficit evidence."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "saved_results/context_e4_ab_area_fusion"
CATEGORIES = ("breakfast_box", "splicing_connectors")


def ab_folder(cat):
    return ROOT / "saved_results" / ("breakfast_ab" if cat == "breakfast_box" else "ab_" + cat)


def calibrate(cat):
    import torch
    import yaml
    from easydict import EasyDict
    from breakfast_ab_inference import (LogiCo, LOCODatasetWithSeg, get_data_transforms,
        load_dinov3_model, get_dinov3_weight_name, setup_seed)
    setup_seed(1)
    config = EasyDict(yaml.safe_load((ROOT / "configs/train/mvtec_loco.yaml").read_text()))
    prompts = yaml.safe_load((ROOT / f"configs/segmentations/mvtec_loco/{cat}.yaml").read_text())["prompts"]
    info = json.loads((ab_folder(cat) / "inference.json").read_text())
    ckpt = Path(info["checkpoint"])
    with ckpt.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == info["checkpoint_sha256"]
    transform, gt_transform = get_data_transforms(config.image_size)
    ds = LOCODatasetWithSeg(str(ROOT / "data/mvtec_loco" / cat), config.image_size, transform, gt_transform, "validation")
    encoder = load_dinov3_model(config.encoder, pretrained_weight_path=str(ROOT/"ckpt"/get_dinov3_weight_name(config.encoder))).cuda().eval()
    encoder.name = config.encoder
    ds.extract_feats(encoder, config.encoder_target_layers, config.encoder_feat_stride, config.encoder_feat_dim,
                     device="cuda:0", use_aggregate_features=config.use_aggregate_features, use_cache=True)
    model = LogiCo(**config.model, encoder=encoder, comp_num=len(prompts), encoder_feat_dim=config.encoder_feat_dim, device="cuda:0")
    model.load_ckpt(str(ckpt))
    model.eval()
    arrays = []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(ds,batch_size=2,shuffle=False,num_workers=0):
            assert all(Path(p).parent.name=="good" for p in batch["image_path"])
            arrays.append(torch.nn.functional.interpolate(model(batch),size=config.resize_mask,mode="bilinear",align_corners=False).cpu().numpy())
    q95,q99=np.quantile(np.concatenate(arrays),[.95,.99])
    assert q99 > q95
    directory=OUT/cat
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"calibration.json").write_text(json.dumps(dict(alpha=q99-q95,q95=q95,q99=q99,images=len(ds),checkpoint_sha256=info["checkpoint_sha256"]),indent=2))



def evaluate(cat, variant, fuse):
    from utils.spro.spro import read_maps, get_auc_spro_results, get_auc_spros_for_metrics
    from utils.spro.image import DefectsConfig
    from utils.spro.aggregation import MetricsAggregator
    from ab_spro_acceleration import validate_and_install, clear

    output = OUT / cat
    alpha = json.loads((output/"calibration.json").read_text())["alpha"]
    source = ab_folder(cat)
    with (source/"scores.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    maps, paths = [], []
    target_ids = set(json.loads((ROOT/"saved_results/context_e4_shared_normal_configuration"/cat/"summary.json").read_text())["target_ids"])
    for row in rows:
        path = Path(row["image_path"])
        with np.load(row["map_path"]) as z:
            heat = z[variant].astype(np.float32)
        if fuse:
            file = ROOT/"saved_results/context_e4_area_full_spro"/cat/"maps"/path.parent.name/(path.stem+".npz")
            with np.load(file) as z:
                evidence = z["new"].astype(np.float32)
            assert heat.shape == evidence.shape == (256,256)
            heat += np.float32(alpha) * evidence
        assert np.isfinite(heat).all()
        paths.append(str(path))
        maps.append(heat)
    cfg = DefectsConfig.create_from_list(json.loads((ROOT/"data/mvtec_loco"/cat/"defects_config.json").read_text()))
    gt, predictions = read_maps(paths, maps, cat, cfg)
    del maps
    validate_and_install(gt, predictions)
    np.random.seed(1)
    metrics = MetricsAggregator(gt_maps=gt, anomaly_maps=predictions, parallel_workers=None, parallel_niceness=20).run(curve_max_distance=.001,max_steps=10)
    overall = get_auc_spro_results(metrics)["auc_spro"]
    score = {k:float(v[.05]) for k,v in overall.items()}
    selected = [p for p in predictions if Path(p.file_path).parent.name in ("good","logical_anomalies")]
    logical = metrics.reduce_to_images(selected)
    defects = {c.defect_config.defect_name for path,g in zip(paths,gt) if Path(path).name in target_ids and Path(path).parent.name=="logical_anomalies" for c in g.channels}
    matched = {Path(path).name for path,g in zip(paths,gt) if Path(path).parent.name=="logical_anomalies" and any(c.defect_config.defect_name in defects for c in g.channels)}
    assert matched == target_ids
    score["target_subset"] = float(get_auc_spros_for_metrics(logical,filter_defect_names_for_spro=defects)[.05])
    name = variant + ("_plus_E" if fuse else "")
    result=dict(category=cat,variant=name,images=len(paths),target_images=len(target_ids),alpha=alpha,au_spro=score)
    if not fuse:
        history=json.loads((source/"spro_results.json").read_text())["variants"][variant]
        for field,k in (("logical_au_spro","logical_anomalies"),("structural_au_spro","structural_anomalies"),("mean_au_spro","mean")):
            assert abs(history[field]-score[k])<1e-6,(history[field],score[k])
    (output/(name+".json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    clear()
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    if len(sys.argv)==2:
        calibrate(sys.argv[1])
    else:
        evaluate(sys.argv[1],sys.argv[2],sys.argv[3]=="fused")
