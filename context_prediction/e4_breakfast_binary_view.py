"""Add an exploratory E > 0 binary panel to the saved breakfast 064 result."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'saved_results/context_e4_breakfast_064'
IMAGE=ROOT/'data/mvtec_loco/breakfast_box/test/logical_anomalies/064.png'
GT=ROOT/'data/mvtec_loco/breakfast_box/ground_truth/logical_anomalies/064/000.png'


def main():
    with np.load(OUT/'breakfast_064.npz') as data:
        expected,observed,evidence=[data[k] for k in ('expected','observed','evidence')]
    if any(x.shape!=(256,256) for x in (expected,observed,evidence)) or not np.isfinite(evidence).all():
        raise ValueError('Unexpected or non-finite saved maps')
    binary=np.uint8(evidence>0)*255
    Image.fromarray(binary).save(OUT/'E_binary_gt_0_256.png')
    size=(512,410)
    with Image.open(IMAGE) as im: original=im.convert('RGB').resize(size,Image.Resampling.BILINEAR)
    with Image.open(GT) as im: gt=im.convert('L').resize(size,Image.Resampling.NEAREST).point(lambda x:255 if x>0 else 0).convert('RGB')
    def color(value):
        heat=cv2.applyColorMap(np.uint8(np.clip(value,0,1)*255),cv2.COLORMAP_INFERNO)
        return Image.fromarray(cv2.cvtColor(cv2.resize(heat,size,interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2RGB))
    binary_display=Image.fromarray(binary).resize(size,Image.Resampling.NEAREST).convert('RGB')
    panels=[(original,'Original'),(gt,'GT'),(color(expected),'Expected occupancy (0-1)'),
            (color(observed),'Observed occupancy'),(color(evidence),'Missing evidence E (0-1)'),
            (binary_display,'E binary: white if E > 0 (not calibrated)')]
    canvas=Image.new('RGB',(1536,868),'white'); draw=ImageDraw.Draw(canvas)
    for i,(panel,label) in enumerate(panels):
        x=(i%3)*512;y=(i//3)*434
        canvas.paste(panel,(x,y+24));draw.text((x+5,y+3),label,fill='black')
    canvas.save(OUT/'breakfast_064_with_binary.png')
    print('Binary rule: E > 0; pixels white:',int((binary>0).sum()),'of',binary.size)


if __name__=='__main__': main()
