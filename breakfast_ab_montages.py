"""Render ranked paired binary maps and unnormalized SMD probabilities."""
import csv
import json
import os
import textwrap
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils.spro.image import DefectsConfig, GroundTruthMap
from utils.spro.spro import IMAGE_SIZE

ROOT=Path(__file__).resolve().parent
CATEGORY=os.environ.get('LOGICO_AB_CATEGORY','breakfast_box')
OUT=ROOT/'saved_results'/('breakfast_ab' if CATEGORY=='breakfast_box' else 'ab_'+CATEGORY)
DEST=OUT/'diagnosis'
DATA=ROOT/'data/mvtec_loco'/CATEGORY
result=json.loads((DEST/'diagnosis.json').read_text(encoding='utf-8'))
cfg=DefectsConfig.create_from_list(json.loads((DATA/'defects_config.json').read_text()))
with (DEST/'region_differences.csv').open(encoding='utf-8',newline='') as f: regions=list(csv.DictReader(f))
thresholds={v:result['variants'][v]['points']['0.01']['threshold'] for v in ('A','B')}

def render_row(axes,entry):
    stem=Path(entry['image']).stem
    with Image.open(DATA/'test/logical_anomalies'/entry['image']) as f: image=np.asarray(f.convert('RGB').resize((384,384)))
    gt=GroundTruthMap.read_from_png_dir(str(DATA/'ground_truth/logical_anomalies'/stem),cfg)
    mask=gt.get_or_over_channels()
    with np.load(OUT/'maps/logical_anomalies'/(stem+'.npz')) as f:
        binary={v:cv2.resize((cv2.resize(f[v],IMAGE_SIZE[CATEGORY],interpolation=cv2.INTER_LINEAR).astype(np.float64)>thresholds[v]).astype(np.uint8),(384,384),interpolation=cv2.INTER_NEAREST) for v in ('A','B')}
        smd=cv2.resize(f['SMD'],(384,384),interpolation=cv2.INTER_LINEAR)
    gt_show=cv2.resize(mask.astype(np.uint8),(384,384),interpolation=cv2.INTER_NEAREST)
    sample=[r for r in regions if r['image']==entry['image']]
    av=np.mean([float(x['spro_A']) for x in sample]); bv=np.mean([float(x['spro_B']) for x in sample])
    for ax in axes:ax.axis('off')
    unique_defects=', '.join(dict.fromkeys(entry['defects'].split(', ')))
    short_defects=textwrap.shorten(unique_defects,width=30,placeholder='…')
    axes[0].imshow(image);axes[0].set_title(entry['image']+'\n'+short_defects,fontsize=8)
    axes[1].imshow(gt_show,cmap='gray',vmin=0,vmax=1);axes[1].set_title('Ground truth',fontsize=9)
    axes[2].imshow(binary['A'],cmap='gray',vmin=0,vmax=1);axes[2].set_title(f'A binary, sPRO {av:.3f}',fontsize=9)
    axes[3].imshow(binary['B'],cmap='gray',vmin=0,vmax=1);axes[3].set_title(f'B binary, sPRO {bv:.3f}',fontsize=9)
    axes[4].imshow(smd,cmap='inferno',vmin=0,vmax=1);axes[4].set_title('SMD raw sigmoid [0,1]',fontsize=9)

for group in ('B_better','A_better'):
    entries=result['rankings'][group]
    if not entries:
        print(f'{group}: no strictly better examples at the chosen work point',flush=True)
        continue
    fig,axes=plt.subplots(len(entries),5,figsize=(15,3*len(entries)),layout='constrained',squeeze=False)
    for idx,entry in enumerate(entries):
        render_row(axes[idx],entry)
        single,axs=plt.subplots(1,5,figsize=(15,3),layout='constrained')
        render_row(axs,entry)
        single.suptitle(f"{group} | delta B-A = {entry['delta']:+.3f} | FPR budget 1%",fontsize=11)
        single.savefig(DEST/(group+'_'+entry['image'].replace('.png','.png')),dpi=130)
        plt.close(single)
    fig.suptitle(f'{group}: ranking at approximately 1% FPR (delta = B - A)',fontsize=15)
    fig.savefig(DEST/(group+'_top5.png'),dpi=140);plt.close(fig)
print('Saved paired montages',flush=True)
if CATEGORY != 'breakfast_box':
    from ab_summary import write_report
    write_report(CATEGORY)
