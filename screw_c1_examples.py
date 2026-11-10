"""Render two requested case checks with one common heatmap scale per sample."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/mvtec_loco/screw_bag'
OUT=ROOT/'saved_results/screw_c1'
for stem in ('006','007'):
    with Image.open(DATA/'test/logical_anomalies'/(stem+'.png')) as image:
        original=np.asarray(image.convert('RGB').resize((384,384)))
    with Image.open(DATA/'ground_truth/logical_anomalies'/stem/'000.png') as mask:
        gt=np.asarray(mask.convert('L').resize((384,384),Image.Resampling.NEAREST))>0
    with np.load(OUT/'maps/logical_anomalies'/(stem+'.npz')) as file:
        maps={name:file[name] for name in ('A','B','C1')}
    low=float(min(x.min() for x in maps.values()))
    high=float(np.percentile(np.concatenate([x.ravel() for x in maps.values()]),99.5))
    fig,axs=plt.subplots(1,5,figsize=(15,3.3),layout='constrained')
    axs[0].imshow(original);axs[0].set_title('Image')
    axs[1].imshow(gt,cmap='gray',vmin=0,vmax=1);axs[1].set_title('Ground truth')
    for ax,name in zip(axs[2:],('A','B','C1')):
        ax.imshow(original);ax.imshow(maps[name],cmap='jet',vmin=low,vmax=high,alpha=.55,extent=(0,384,384,0));ax.set_title(name)
    for ax in axs:ax.axis('off')
    fig.suptitle(f'Logical anomaly {stem}.png: screw too long; shared A/B/C1 color scale')
    fig.savefig(OUT/(stem+'_comparison.png'),dpi=140);plt.close(fig)
print('Saved examples',flush=True)
