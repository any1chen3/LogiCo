"""Export original image-level scores, without changing model checkpoints."""
import argparse
import csv
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from tqdm import tqdm
from dinov3.hub.backbones import load_dinov3_model
from models import LogiCo
from utils.dataset import LOCODatasetWithSeg, get_data_transforms
from utils.utils import setup_seed, get_dinov3_weight_name

ROOT = Path(__file__).resolve().parent

def main(category):
    setup_seed(1)
    cfg = EasyDict(yaml.safe_load((ROOT / 'configs/train/mvtec_loco.yaml').read_text()))
    prompts = yaml.safe_load((ROOT / 'configs/segmentations/mvtec_loco' / (category+'.yaml')).read_text())['prompts']
    transform, gt_transform = get_data_transforms(cfg.image_size)
    data = LOCODatasetWithSeg(root=str(ROOT/'data/mvtec_loco'/category), image_size=cfg.image_size, transform=transform, gt_transform=gt_transform, phase='test')
    device='cuda:0'
    encoder=load_dinov3_model(cfg.encoder, pretrained_weight_path=str(ROOT/'ckpt'/get_dinov3_weight_name(cfg.encoder))).to(device).eval()
    encoder.name=cfg.encoder
    data.extract_feats(encoder, cfg.encoder_target_layers, cfg.encoder_feat_stride, cfg.encoder_feat_dim, device=device, use_aggregate_features=cfg.use_aggregate_features, use_cache=True)
    loader=torch.utils.data.DataLoader(data,batch_size=2,shuffle=False,num_workers=4)
    model=LogiCo(**cfg.model,encoder=encoder,comp_num=len(prompts),encoder_feat_dim=cfg.encoder_feat_dim,device=device)
    results='saved_results_restart' if category=='juice_bottle' else 'saved_results'
    model.load_ckpt(str(ROOT/results/'mvtec_loco'/category/'LogiCo_best_ckpt.pkl'))
    model.eval()
    rows=[]
    with torch.no_grad():
        for batch in tqdm(loader,desc=category):
            maps=F.interpolate(model(batch),size=cfg.resize_mask,mode='bilinear',align_corners=False)
            n=int(cfg.resize_mask*cfg.resize_mask*cfg.max_ratio)
            scores=torch.sort(maps.flatten(1),dim=1,descending=True)[0][:,:n].mean(1).cpu().numpy()
            for i,path in enumerate(batch['image_path']):
                rel=Path(path).relative_to(ROOT/'data/mvtec_loco'/category/'test')
                rows.append(dict(image=str(rel).replace('\\','/'),score=float(scores[i])))
    folder=ROOT/'saved_results'/'image_scores';folder.mkdir(parents=True,exist_ok=True)
    temp=folder/(category+'.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['image','score']);w.writeheader();w.writerows(rows)
    temp.replace(folder/(category+'.csv'))
    print(f'{category}: saved {len(rows)} scores',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('category');main(parser.parse_args().category)
