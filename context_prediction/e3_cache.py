"""Build E3 normal-only RGB tile and SAM component caches, resumably."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from context_prediction.proposal_audit import ROOT, WEIGHT, EXPECTED_SHA
from context_prediction.fast_mask_selection import select_masks_fast
from context_prediction.tile_isolation import encode_tiles, MEAN, STD
from dinov3.hub.backbones import load_dinov3_model
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

OUT = ROOT/'saved_results/context_e3/cache_v1'
DINO = ROOT/'ckpt/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--categories',nargs='+',default=['splicing_connectors','juice_bottle'])
    ap.add_argument('--splits',nargs='+',default=['train','validation'])
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    if digest(WEIGHT)!=EXPECTED_SHA: raise RuntimeError('SAM checkpoint hash mismatch')
    OUT.mkdir(parents=True,exist_ok=True)
    config=dict(version=1,sam_sha256=EXPECTED_SHA,dino_sha256=digest(DINO),grid=32,points_per_batch=8,
                predicted_iou_min=.8,stability_min=.9,max_side=512,area_min=16,area_max=.8,
                duplicate_iou=.85,tile_size=224,crop_size=224,dino_batch=4)
    meta=OUT/'config.json'
    if meta.exists() and json.loads(meta.read_text())!=config: raise RuntimeError('Cache settings changed: use new cache version')
    meta.write_text(json.dumps(config,indent=2),encoding='utf-8')
    device='cuda:0'
    dino=load_dinov3_model('dinov3_vitb16',pretrained_weight_path=str(DINO)).to(device).eval()
    sam=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(WEIGHT),device=device,apply_postprocessing=False)
    gen=SAM2AutomaticMaskGenerator(sam,points_per_side=32,points_per_batch=8,pred_iou_thresh=.8,
                                   stability_score_thresh=.9,box_nms_thresh=1.,min_mask_region_area=0)
    for category in args.categories:
        for split in args.splits:
            paths=sorted((ROOT/'data/mvtec_loco'/category/split/'good').glob('*.png'))
            if args.limit: paths=paths[:args.limit]
            for number,path in enumerate(paths,1):
                dest=OUT/category/split/(path.stem+'.npz')
                src_hash=digest(path)
                if dest.exists():
                    with np.load(dest) as saved:
                        if str(saved['source_sha256'])!=src_hash: raise RuntimeError('Cached input changed: '+str(path))
                    print('cached',category,split,number,'/',len(paths),path.name,flush=True)
                    continue
                started=time.perf_counter()
                with Image.open(path) as im: source=im.convert('RGB')
                tile_feat,tile_lab=encode_tiles(source,dino,device)
                scale=512/max(source.size)
                resized=source.resize((round(source.width*scale),round(source.height*scale)),Image.Resampling.BILINEAR)
                rgb=np.array(resized); h,w=rgb.shape[:2]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): proposals=gen.generate(rgb)
                selected=select_masks_fast(proposals,h,w)
                lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32); lab[...,0]*=100/255; lab[...,1:]-=128
                masks=[]; centers=[]; areas=[]; colors=[]; crops=[]; qualities=[]
                for p,idx,area,bbox in selected:
                    mask=p['segmentation']; ys,xs=np.nonzero(mask)
                    if not len(xs): continue
                    x1,x2=int(xs.min()),int(xs.max())+1; y1,y2=int(ys.min()),int(ys.max())+1
                    crop=Image.fromarray(rgb[y1:y2,x1:x2]).resize((224,224),Image.Resampling.BICUBIC)
                    crops.append(TF.normalize(TF.to_tensor(crop),MEAN,STD)); masks.append(mask)
                    centers.append([(float(xs.mean())+.5)/w,(float(ys.mean())+.5)/h])
                    areas.append(float(mask.sum())/(h*w)); colors.append(np.median(lab[mask],axis=0))
                    qualities.append([p['predicted_iou'],p['stability_score']])
                features=[]
                with torch.inference_mode():
                    for i in range(0,len(crops),4):
                        batch=torch.stack(crops[i:i+4]).to(device)
                        features.append(F.normalize(dino.forward_features(batch)['x_norm_clstoken'],dim=1).cpu().numpy())
                feat=np.concatenate(features) if features else np.zeros((0,768),np.float32)
                dest.parent.mkdir(parents=True,exist_ok=True)
                temporary=dest.with_suffix('.tmp.npz')
                np.savez_compressed(temporary,source_sha256=src_hash,source_path=str(path),source_size=source.size,
                                    tile_features=tile_feat,tile_lab=tile_lab,region_features=feat,
                                    region_centers=np.asarray(centers,dtype=np.float32).reshape(-1,2),
                                    region_area=np.asarray(areas,dtype=np.float32),region_lab=np.asarray(colors,dtype=np.float32).reshape(-1,3),
                                    region_quality=np.asarray(qualities,dtype=np.float32).reshape(-1,2),
                                    masks=np.asarray(masks,dtype=bool).reshape(-1,h,w))
                temporary.replace(dest)
                print(json.dumps(dict(category=category,split=split,position=number,total=len(paths),image=path.name,
                                      proposals=len(proposals),regions=len(masks),seconds=round(time.perf_counter()-started,1),
                                      peak_mib=round(torch.cuda.max_memory_allocated()/1024**2,1))),flush=True)


if __name__=='__main__': main()
