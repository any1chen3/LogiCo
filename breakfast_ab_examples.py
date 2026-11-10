"""Save paired heatmap examples using a shared color scale per image."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/mvtec_loco/breakfast_box'
OUT=ROOT/'saved_results/breakfast_ab'
for filename, title in [('024.png','Missing bananas'), ('068.png','Four tangerines')]:
    with Image.open(DATA/'test/logical_anomalies'/filename) as f:
        image=np.asarray(f.convert('RGB').resize((384,384)))
    with Image.open(DATA/'ground_truth/logical_anomalies'/Path(filename).stem/'000.png') as f:
        gt=np.asarray(f.convert('L').resize((384,384),Image.Resampling.NEAREST))>0
    with np.load(OUT/'maps/logical_anomalies'/(Path(filename).stem+'.npz')) as data:
        a=data['A'];b=data['B']
    lo=float(min(a.min(),b.min()));hi=float(np.percentile(np.concatenate([a.ravel(),b.ravel()]),99.5))
    fig,axes=plt.subplots(1,4,figsize=(14,3.6),layout='constrained')
    for ax in axes: ax.axis('off')
    axes[0].imshow(image);axes[0].set_title('Image')
    for ax, m, name in [(axes[1],a,'A: CRN + SRN + SMD'),(axes[2],b,'B: CRN + SRN')]:
        ax.imshow(image);ax.imshow(m,cmap='jet',vmin=lo,vmax=hi,alpha=.55,extent=(0,384,384,0));ax.set_title(name)
    axes[3].imshow(gt,cmap='gray',vmin=0,vmax=1);axes[3].set_title('Ground truth')
    fig.suptitle(title+' — '+filename+' (A/B share color scale)')
    fig.savefig(OUT/(Path(filename).stem+'_AB.png'),dpi=140);plt.close(fig)
