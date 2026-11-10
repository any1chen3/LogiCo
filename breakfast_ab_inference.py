"""Export original branch outputs and paired maps with one forward per batch."""
import csv
import hashlib
import json
import os
from pathlib import Path
import types
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from tqdm import tqdm
from dinov3.hub.backbones import load_dinov3_model
from models import LogiCo
from utils.dataset import LOCODatasetWithSeg, get_data_transforms
from utils.metrics import compute_imagewise_log_metrics
from utils.utils import setup_seed, get_dinov3_weight_name

ROOT = Path(__file__).resolve().parent
CATEGORY = os.environ.get('LOGICO_AB_CATEGORY', 'breakfast_box')
OUT = ROOT / 'saved_results' / ('breakfast_ab' if CATEGORY == 'breakfast_box' else 'ab_' + CATEGORY)
SAVE = 'saved_results_restart' if CATEGORY == 'juice_bottle' else 'saved_results'
CKPT = ROOT / SAVE / 'mvtec_loco' / CATEGORY / 'LogiCo_best_ckpt.pkl'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'maps').mkdir(exist_ok=True)
    setup_seed(1)
    cfg = EasyDict(yaml.safe_load((ROOT / 'configs/train/mvtec_loco.yaml').read_text()))
    prompts = yaml.safe_load((ROOT / 'configs/segmentations/mvtec_loco' / (CATEGORY+'.yaml')).read_text())['prompts']
    transform, gt_transform = get_data_transforms(cfg.image_size)
    data = LOCODatasetWithSeg(root=str(ROOT / 'data/mvtec_loco' / CATEGORY), image_size=cfg.image_size, transform=transform, gt_transform=gt_transform, phase='test')
    device = 'cuda:0'
    encoder = load_dinov3_model(cfg.encoder, pretrained_weight_path=str(ROOT / 'ckpt' / get_dinov3_weight_name(cfg.encoder))).to(device).eval()
    encoder.name = cfg.encoder
    data.extract_feats(encoder, cfg.encoder_target_layers, cfg.encoder_feat_stride, cfg.encoder_feat_dim, device=device, use_aggregate_features=cfg.use_aggregate_features, use_cache=True)
    loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False, num_workers=4)
    model = LogiCo(**cfg.model, encoder=encoder, comp_num=len(prompts), encoder_feat_dim=cfg.encoder_feat_dim, device=device)
    model.load_ckpt(str(CKPT))
    model.eval()
    original_fusion = model.anomaly_map_fusion
    capture = {}
    def fusion(self, crn, srn, smd, output_size):
        capture['branches'] = (crn.detach(), srn.detach(), smd.detach())
        capture['B'] = original_fusion(crn, srn, None, output_size)
        return original_fusion(crn, srn, smd, output_size)
    model.anomaly_map_fusion = types.MethodType(fusion, model)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='A/B inference'):
            a = model(batch)
            b = capture['B']
            a = F.interpolate(a, size=cfg.resize_mask, mode='bilinear', align_corners=False)
            b = F.interpolate(b, size=cfg.resize_mask, mode='bilinear', align_corners=False)
            n = int(cfg.resize_mask * cfg.resize_mask * cfg.max_ratio)
            scores = torch.sort(a.flatten(1), dim=1, descending=True)[0][:, :n].mean(dim=1).cpu().numpy()
            raw = [x.cpu().numpy() for x in capture['branches']]
            for i, path in enumerate(batch['image_path']):
                rel = Path(path).relative_to(ROOT / 'data/mvtec_loco' / CATEGORY / 'test')
                destination = OUT / 'maps' / rel.parent / (rel.stem + '.npz')
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, CRN=raw[0][i,0], SRN=raw[1][i,0], SMD=raw[2][i,0], A=a[i,0].cpu().numpy(), B=b[i,0].cpu().numpy())
                rows.append(dict(image_path=str(path), map_path=str(destination), defect_type=rel.parent.name, label=int(batch['label'][i]), score_A=float(scores[i])))
    with (OUT / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    metrics = compute_imagewise_log_metrics([r['score_A'] for r in rows], [r['label'] for r in rows], [r['image_path'] for r in rows])
    with CKPT.open('rb') as f:
        sha=hashlib.file_digest(f, 'sha256').hexdigest()
    summary = dict(checkpoint=str(CKPT), checkpoint_sha256=sha, batch_size=2, seed=1, count=len(rows), image_metrics=metrics, image_score='A top 0.5% mean after 256x256 interpolation', maps='native raw CRN/SRN/SMD; A/B gaussian-smoothed then interpolated 256x256', selected_on_test=True)
    (OUT / 'inference.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
