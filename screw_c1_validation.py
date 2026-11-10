"""Calibrate C1 only on normal validation pixels; no test maps are read."""
import hashlib,json,types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from tqdm import tqdm
from dinov3.hub.backbones import load_dinov3_model
from models import LogiCo
from utils.dataset import LOCODatasetWithSeg,get_data_transforms
from utils.utils import setup_seed,get_dinov3_weight_name

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'saved_results/screw_c1'
CKPT=ROOT/'saved_results/mvtec_loco/screw_bag/LogiCo_best_ckpt.pkl'

def main():
    OUT.mkdir(parents=True,exist_ok=True);setup_seed(1)
    cfg=EasyDict(yaml.safe_load((ROOT/'configs/train/mvtec_loco.yaml').read_text()))
    prompts=yaml.safe_load((ROOT/'configs/segmentations/mvtec_loco/screw_bag.yaml').read_text())['prompts']
    transform,gt_transform=get_data_transforms(cfg.image_size)
    dataset=LOCODatasetWithSeg(root=str(ROOT/'data/mvtec_loco/screw_bag'),image_size=cfg.image_size,transform=transform,gt_transform=gt_transform,phase='validation')
    if not dataset.samples or any(Path(s[0]).parent.name!='good' for s in dataset.samples):raise RuntimeError('Validation inputs must contain only validation/good')
    encoder=load_dinov3_model(cfg.encoder,pretrained_weight_path=str(ROOT/'ckpt'/get_dinov3_weight_name(cfg.encoder))).cuda().eval();encoder.name=cfg.encoder
    dataset.extract_feats(encoder,cfg.encoder_target_layers,cfg.encoder_feat_stride,cfg.encoder_feat_dim,device='cuda:0',use_aggregate_features=cfg.use_aggregate_features,use_cache=True)
    loader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False,num_workers=4)
    model=LogiCo(**cfg.model,encoder=encoder,comp_num=len(prompts),encoder_feat_dim=cfg.encoder_feat_dim,device='cuda:0')
    model.load_ckpt(str(CKPT));model.eval()
    original=model.anomaly_map_fusion;values={'R':[],'SMD':[]}
    def fusion(self,crn,srn,smd,output_size):
        size=(output_size,output_size) if isinstance(output_size,int) else output_size
        def norm(x,mu,std):return (F.interpolate(x,size=size,mode='bilinear',align_corners=True)-mu)/std
        c=norm(crn,self.CRN_mean,self.CRN_std);r=norm(srn,self.SRN_mean,self.SRN_std);s=norm(smd,self.SMD_mean,self.SMD_std)
        values['R'].append(torch.maximum(c,r).flatten().cpu().numpy())
        values['SMD'].append(s.flatten().cpu().numpy())
        return original(crn,srn,smd,output_size)
    model.anomaly_map_fusion=types.MethodType(fusion,model)
    with torch.no_grad():
        for batch in tqdm(loader,desc='Normal validation'):
            model(batch)
    r=np.concatenate(values['R']);s=np.concatenate(values['SMD'])
    r95,r99=map(float,np.percentile(r,[95,99]));s95,s99=map(float,np.percentile(s,[95,99]))
    gap=s99-s95;tolerance=1e-6*max(1.0,abs(s95),abs(s99))
    with CKPT.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    result={'category':'screw_bag','split':'validation/good','validation_image_count':len(dataset),'pixels_per_branch':len(r),'resolution':[cfg.image_size,cfg.image_size],'percentile_method':'numpy linear on pooled pixels, standardized and interpolated, before fusion and Gaussian smoothing','r95':r95,'r99':r99,'s95':s95,'s99':s99,'smd_gap':gap,'gap_tolerance':tolerance,'checkpoint_sha256':sha,'batch_size':2}
    if not np.isfinite([r95,r99,s95,s99]).all() or gap<=tolerance or r99<=r95:
        result['status']='invalid_percentiles';(OUT/'calibration.json').write_text(json.dumps(result,indent=2),encoding='utf-8');raise ValueError('Invalid or near-zero percentile interval; inspect outputs before testing')
    result['scale']=(r99-r95)/gap;result['offset']=r95-result['scale']*s95;result['status']='valid'
    (OUT/'calibration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
