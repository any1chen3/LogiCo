"""Paired LOCO sPRO/FPR curves and region diagnostics at equal FP budgets."""
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils.spro.image import DefectsConfig, GroundTruthMap
from utils.spro.spro import IMAGE_SIZE

ROOT = Path(__file__).resolve().parent
CATEGORY = os.environ.get('LOGICO_AB_CATEGORY', 'breakfast_box')
OUT = ROOT / 'saved_results' / ('breakfast_ab' if CATEGORY == 'breakfast_box' else 'ab_' + CATEGORY)
DEST = OUT / 'diagnosis'
DATA = ROOT / 'data/mvtec_loco' / CATEGORY
SIZE = IMAGE_SIZE[CATEGORY]
GRID = np.linspace(0, .05, 201)
NBINS = 65536

def gt_for(row,cfg):
    if row['defect_type'] == 'good': return None
    folder=DATA/'ground_truth'/row['defect_type']/Path(row['image_path']).stem
    return GroundTruthMap.read_from_png_dir(png_dir=str(folder), defects_config=cfg)

def maps_for(row):
    with np.load(row['map_path']) as f:
        return {v:cv2.resize(f[v],SIZE,interpolation=cv2.INTER_LINEAR) for v in ('A','B')}

def kind(name):
    if CATEGORY != 'breakfast_box':
        if 'missing' in name or 'not_enough' in name or name.startswith('empty_'): return '缺失或不足'
        if 'additional' in name or name.startswith('extra_') or 'too_much' in name: return '多件或过量'
        if any(x in name for x in ('swapped','misplaced','location','not_plugged','flipped','rotated')): return '错位或连接异常'
        return '其他类型（按原始名称）'
    if name.startswith('missing_') or name in ('1_nectarine_1_tangerine','0_nectarines_1_tangerine','0_nectarines_0_tangerines','underflow'): return '缺件或不足'
    if name in ('0_nectarines_4_tangerines','overflow'): return '多件或溢出'
    if name in ('compartments_swapped','2_nectarines_1_tangerine','0_nectarines_2_tangerines','0_nectarines_3_tangerines','3_nectarines_0_tangerines','wrong_ratio','mixed_cereals'): return '错放或成分异常'
    return '其他逻辑异常'

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    with (OUT/'scores.csv').open(encoding='utf-8',newline='') as f: rows=[r for r in csv.DictReader(f) if r['defect_type'] in ('good','logical_anomalies')]
    cfg=DefectsConfig.create_from_list(json.loads((DATA/'defects_config.json').read_text()))
    lo={v:np.inf for v in ('A','B')}; hi={v:-np.inf for v in ('A','B')}
    for row in rows:
        with np.load(row['map_path']) as f:
            for v in lo: lo[v]=min(lo[v],float(f[v].min()));hi[v]=max(hi[v],float(f[v].max()))
    edges={v:np.linspace(lo[v]-1e-5,hi[v]+1e-5,NBINS+1) for v in lo}
    hist={v:np.zeros(NBINS,dtype=np.int64) for v in lo}
    print('Phase 1: building background-score histograms',flush=True)
    for i,row in enumerate(rows):
        gt=gt_for(row,cfg); bg=None if gt is None else ~gt.get_or_over_channels(); maps=maps_for(row)
        for v,m in maps.items(): hist[v]+=np.histogram(m.ravel() if bg is None else m[bg],bins=edges[v])[0]
        if (i+1)%25==0: print(f'histograms {i+1}/{len(rows)}',flush=True)
    thresholds={}
    for v in lo:
        reverse=np.r_[0,np.cumsum(hist[v][::-1])]; total=reverse[-1]
        positions=np.searchsorted(reverse,GRID*total,side='right')-1
        positions=np.clip(positions,0,NBINS)
        thresholds[v]=edges[v][NBINS-positions]
    print('Phase 2: measuring exact rates at candidate thresholds',flush=True)
    fps={v:np.zeros(len(GRID),np.int64) for v in lo}; backgrounds=0
    channels=[]
    for i,row in enumerate(rows):
        gt=gt_for(row,cfg); maps=maps_for(row)
        bg=None if gt is None else ~gt.get_or_over_channels()
        backgrounds += maps['A'].size if bg is None else int(bg.sum())
        for v,m in maps.items():
            values=np.sort(m.ravel() if bg is None else m[bg])
            fps[v] += len(values)-np.searchsorted(values,thresholds[v],side='right')
        if gt is not None:
            for j,channel in enumerate(gt.channels):
                per={}
                for v,m in maps.items():
                    values=np.sort(m[channel.bool_array]);count=len(values)-np.searchsorted(values,thresholds[v],side='right')
                    per[v]=np.minimum(count/channel.get_saturation_area(),1.0)
                name=channel.defect_config.defect_name
                channels.append(dict(image=Path(row['image_path']).name, channel=j, defect_name=name, type=kind(name), saturation_area=int(channel.get_saturation_area()), A=per['A'], B=per['B']))
        if (i+1)%25==0: print(f'measured {i+1}/{len(rows)}',flush=True)
    result={'protocol':{'category':CATEGORY,'scope':'logical_anomalies + good; FPR on all pixels outside union of GT channels','threshold_rule':'prediction > threshold; choose smallest threshold with actual FPR <= budget from histogram candidates','curve':'201 target budgets with 65536-bin background histogram; measured exact FPR and per-channel sPRO at resulting thresholds','ground_truth':'all channels; original saturation areas','fpr_denominator':backgrounds,'image_ranking':'mean of per-region differences per image; sort and select five distinct images per side','subset_type_mapping':'descriptive grouping from names; no inferred cause'},'variants':{}}
    curves={}
    for v in ('A','B'):
        rates=fps[v]/backgrounds
        avg=np.mean([r[v] for r in channels],axis=0)
        indices={str(x):int(np.where(rates<=x+1e-15)[0][-1]) for x in (.01,.03,.05)}
        points={k:dict(target_fpr=float(k), threshold=float(thresholds[v][idx]), actual_fpr=float(rates[idx]), spro=float(avg[idx]),index=idx) for k,idx in indices.items()}
        result['variants'][v]={'points':points}
        curves[v]=(rates,avg,thresholds[v])
    with (DEST/'curve_points.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['variant','target_fpr','actual_fpr','threshold','mean_spro'])
        for v in ('A','B'):
            for target,rate,thr,mean in zip(GRID,*[curves[v][0],curves[v][2],curves[v][1]]):writer.writerow([v,target,rate,thr,mean])
    ia=result['variants']['A']['points']['0.01']['index'];ib=result['variants']['B']['points']['0.01']['index']
    records=[]
    for r in channels:
        x={k:r[k] for k in ('image','channel','defect_name','type','saturation_area')};x.update(spro_A=float(r['A'][ia]),spro_B=float(r['B'][ib]),delta=float(r['B'][ib]-r['A'][ia]));records.append(x)
    with (DEST/'region_differences.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    byimage=defaultdict(list)
    for r in records:byimage[r['image']].append(r)
    ranked=sorted([dict(image=k,delta=float(np.mean([q['delta'] for q in vals])),regions=len(vals),defects=', '.join(q['defect_name'] for q in vals),types=', '.join(sorted(set(q['type'] for q in vals)))) for k,vals in byimage.items()],key=lambda x:x['delta'])
    result['rankings']={'A_better':[r for r in ranked if r['delta'] < -1e-12][:5], 'B_better':[r for r in reversed(ranked) if r['delta'] > 1e-12][:5]}
    (DEST/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for v in ('A','B'):ax.plot(curves[v][0]*100,curves[v][1]*100,label=v,linewidth=2)
    ax.set(xlim=(0,5),ylim=(0,100),xlabel='False positive rate (%)',ylabel='Logical sPRO (%)',title=CATEGORY+': logical sPRO vs FPR');ax.grid(alpha=.25);ax.legend();fig.savefig(DEST/'spro_fpr_curves.png',dpi=180);plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    import subprocess, sys
    subprocess.run([sys.executable, '-u', str(ROOT / 'breakfast_ab_verify_points.py')], cwd=ROOT, check=True)

if __name__=='__main__':main()
