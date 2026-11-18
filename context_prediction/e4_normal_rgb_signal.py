"""Check fixed RGB colour signal on normal train/validation; not a wire detector."""
import csv,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/mvtec_loco/splicing_connectors'
OUT=ROOT/'saved_results/context_e4_normal_rgb_signal'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split in ('train/good','validation/good'):
        files=sorted((DATA/split).glob('*.png'))
        for path in files:
            with Image.open(path) as im: rgb=np.asarray(im.convert('RGB').resize((512,256),Image.Resampling.BILINEAR))
            hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
            color=(hsv[:,:,1]>=120)&(hsv[:,:,2]>=60)
            band=color[:,230:282]
            rows.append(dict(split=split,image=path.name,center_color_fraction=float(band.mean()),center_color_pixels=int(band.sum())))
    train=np.array([r['center_color_fraction'] for r in rows if r['split']=='train/good'])
    val=np.array([r['center_color_fraction'] for r in rows if r['split']=='validation/good'])
    if len(train)!=360 or len(val)!=60:raise ValueError('Unexpected split sizes')
    # Predefined exploration: lower empirical 5th percentile on training only.
    cutoff=float(np.quantile(train,.05,method='lower'))
    for row in rows:row['below_train_cutoff']=bool(row['center_color_fraction']<cutoff)
    with (OUT/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    def stats(a):return dict(n=len(a),min=float(a.min()),p05=float(np.quantile(a,.05)),median=float(np.median(a)),max=float(a.max()),below_train_cutoff=int((a<cutoff).sum()))
    summary=dict(protocol='Exploratory RGB only, 512x256 fixed center strip x=[230,282), saturation>=120 and value>=60 on OpenCV HSV. Cutoff is lower empirical 5th percentile on 360 normal training images; validation uses fixed cutoff. All images are normal; no wire absence labels or new inference. Image resizing uses bilinear PIL.',
      warning='Color in strip is not necessarily wire; strip includes connector and background. This is normal false-warning analysis only, no sensitivity, specificity for missing wire, or deployment performance. Prior test exploratory figures used PIL default resampling, so do not compare exact scores without harmonization.',
      cutoff=cutoff,train=stats(train),validation=stats(val))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
