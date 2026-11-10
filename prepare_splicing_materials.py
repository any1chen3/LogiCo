"""Prepare per-image evidence and representative samples for LOCO connectors."""
import csv,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from utils.spro.image import DefectsConfig,GroundTruthMap
from utils.spro.spro import IMAGE_SIZE

ROOT=Path(__file__).resolve().parent
CATEGORY='splicing_connectors'
DATA=ROOT/'data/mvtec_loco'/CATEGORY
SEG=ROOT/'data/mvtec_loco_segmentations'/CATEGORY/'test'
AB=ROOT/'saved_results/ab_splicing_connectors'
DEST=ROOT/'saved_results/splicing_connectors_materials'
SIZE=IMAGE_SIZE[CATEGORY]
selection=[('both_poor','015.png'),('both_poor','098.png'),('both_poor','026.png'),('both_poor','075.png'),('A_better','056.png'),('A_better','006.png'),('B_better','067.png'),('B_better','102.png')]

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    diag=json.loads((AB/'diagnosis/diagnosis.json').read_text(encoding='utf-8'))
    points={v:diag['variants'][v]['points']['0.01'] for v in ('A','B')}
    cfg=DefectsConfig.create_from_list(json.loads((DATA/'defects_config.json').read_text()))
    with (AB/'scores.csv').open(encoding='utf-8',newline='') as f: source=list(csv.DictReader(f))
    if len(source)!=312:raise ValueError('Expected 312 test images')
    with (AB/'diagnosis/region_differences.csv').open(encoding='utf-8',newline='') as f:
        previous={(r['image'],int(r['channel'])):r for r in csv.DictReader(f)}
    records=[];region_records=[]
    for row in source:
        name=Path(row['image_path']).name;kind=row['defect_type'];gt=None
        gtfiles=[]
        if kind!='good':
            gtpath=DATA/'ground_truth'/kind/Path(name).stem
            gtfiles=sorted(gtpath.glob('*.png'))
            gt=GroundTruthMap.read_from_png_dir(str(gtpath),cfg)
        vals={'A':[],'B':[]};names=[]
        if gt:
            with np.load(row['map_path']) as maps:
                binary={v:cv2.resize(maps[v],SIZE,interpolation=cv2.INTER_LINEAR).astype(np.float64)>points[v]['threshold'] for v in ('A','B')}
            for j,ch in enumerate(gt.channels):
                n=ch.defect_config.defect_name;names.append(n)
                for v in ('A','B'):vals[v].append(float(min(np.count_nonzero(binary[v]&ch.bool_array)/ch.get_saturation_area(),1.0)))
                if kind=='logical_anomalies':
                    old=previous[(name,j)]
                    for v in ('A','B'):
                        if abs(vals[v][-1]-float(old['spro_'+v]))>1e-12:raise AssertionError(f'Logical-region mismatch: {name}/{j}/{v}')
                region_records.append(dict(image_id=kind+'/'+name,annotation_file=gtfiles[j].name,defect_name=n,saturation_area=int(ch.get_saturation_area()),spro_A=vals['A'][-1],spro_B=vals['B'][-1],delta_B_minus_A=vals['B'][-1]-vals['A'][-1]))
        record=dict(image_id=kind+'/'+name,image_number=Path(name).stem,image_type=kind,defect_names='; '.join(dict.fromkeys(names)),annotation_count=len(names),image_score_A=float(row['score_A']),spro_A=float(np.mean(vals['A'])) if vals['A'] else None,spro_B=float(np.mean(vals['B'])) if vals['B'] else None,delta_B_minus_A=float(np.mean(vals['B'])-np.mean(vals['A'])) if vals['A'] else None)
        records.append(record)
    fields=list(records[0]);
    with (DEST/'逐图结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    with (DEST/'逐标注区域.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(region_records[0]));w.writeheader();w.writerows(region_records)
    thresholds={'target_fpr':0.01,'comparison_scope':'good + logical_anomalies','negative_pixels':diag['protocol']['fpr_denominator'],'selection':'per variant, score > threshold, candidate threshold with actual FPR <= 0.01','A':points['A'],'B':points['B'],'structural_note':'同一对阈值也应用于结构异常图，但实际 FPR 是在正常+逻辑异常集合上计算的，并非结构异常集合的 FPR。'}
    (DEST/'工作点与阈值.json').write_text(json.dumps(thresholds,ensure_ascii=False,indent=2),encoding='utf-8')
    byid={r['image_id']:r for r in records};normal=sorted([r for r in records if r['image_type']=='good'],key=lambda r:r['image_score_A'])
    normal_cases=[normal[len(normal)//2],normal[min(len(normal)-1,int(.95*(len(normal)-1)))]]
    chosen=[('normal_median',normal_cases[0]['image_id']),('normal_high_score_p95',normal_cases[1]['image_id'])]
    chosen=[(why,'logical_anomalies/'+name) for why,name in selection]+chosen
    assert len({key for _,key in chosen})==10
    config=yaml.safe_load((ROOT/'configs/segmentations/mvtec_loco/splicing_connectors.yaml').read_text())
    names=config['prompts'];palette=np.array(config['palette'][:3*len(names)],dtype=np.uint8).reshape(-1,3)
    cases=[]
    for reason,key in chosen:
        kind,name=key.split('/');item=byid[key];image=DATA/'test'/kind/name;seg=SEG/kind/name;srcmap=AB/'maps'/kind/(Path(name).stem+'.npz')
        folder=DEST/'代表样本'/(reason+'__'+kind+'__'+Path(name).stem);folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(image,folder/'原图.png');shutil.copy2(seg,folder/'组件分割_原始文件.png');shutil.copy2(srcmap,folder/'原始分支与融合图.npz')
        gt=DATA/'ground_truth'/kind/Path(name).stem
        if gt.exists():
            outgt=folder/'GT_原始标注';outgt.mkdir(exist_ok=True)
            for path in sorted(gt.glob('*.png')):shutil.copy2(path,outgt/path.name)
        with Image.open(seg) as im:seg512=np.asarray(im.resize((512,512),Image.Resampling.NEAREST)).copy()
        actual=F.interpolate(torch.from_numpy(seg512.astype(np.float32))[None,None],size=(256,256),mode='nearest')[0,0].numpy().astype(np.uint8)
        if actual.max()>=len(names):raise ValueError(f'Unexpected segmentation id for {key}')
        Image.fromarray(actual).save(folder/'SMD输入_类别索引_256.png')
        Image.fromarray(palette[actual]).save(folder/'SMD输入_颜色图_256.png')
        one_hot=np.moveaxis(np.eye(len(names),dtype=np.float32)[actual],-1,0)
        np.savez_compressed(folder/'SMD输入_onehot_256.npz',onehot=one_hot)
        case={'selection_reason':reason,'image_id':key,'record':item,'folder':str(folder.relative_to(DEST)),'smd_input':{'size':[256,256],'class_names':names,'onehot_shape':list(one_hot.shape),'construction':'source seg -> PIL nearest 512 -> torch nearest 256 -> one-hot','palette_rgb':palette.tolist()},'arrays':{'CRN':'raw cosine distance, 64x64, before standardization','SRN':'raw cosine distance, 64x64, before standardization','SMD':'raw sigmoid output, 256x256, before standardization','A':'original fused and Gaussian-smoothed map, resized 256x256','B':'CRN/SRN fused and Gaussian-smoothed map, resized 256x256'}}
        (folder/'样本说明.json').write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8');cases.append(case)
    (DEST/'代表样本清单.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    print('images',len(records),'regions',len(region_records),'types',dict(Counter(r['image_type'] for r in records)),'cases',[(x['selection_reason'],x['image_id']) for x in cases],flush=True)

if __name__=='__main__':main()
