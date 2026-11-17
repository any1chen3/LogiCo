"""Audit fixed full-height bands on separate normal validation images only."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/mvtec_loco/splicing_connectors/validation/good'
SEG=ROOT/'data/mvtec_loco_segmentations/splicing_connectors/validation/good'
OUT=ROOT/'saved_results/context_e4_fixed_band_validation_audit'
# Chosen before checking 010-029. Full-height bands; all coordinates on 256 grid.
BANDS=[('previous',55,207),('wide',48,208),('middle',64,192),('narrow',80,176)]


def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'examples').mkdir(exist_ok=True)
    images=sorted(DATA.glob('*.png'))[10:30]
    if len(images)!=20 or images[0].stem!='010' or images[-1].stem!='029':raise ValueError('Expected validation 010–029')
    records=[]; examples={'010','015','020'}
    for path in images:
        with Image.open(SEG/path.name) as im:
            if im.mode not in ('P','L'):raise ValueError('Expected label image: '+str(path))
            labels=np.asarray(im.resize((256,256),Image.Resampling.NEAREST))
        if labels.max()>4:raise ValueError('Unexpected segmentation label')
        line=np.isin(labels,[0,1,2]); total=int(line.sum())
        num,components,stats,centers=cv2.connectedComponentsWithStats(np.uint8(labels==3),8)
        con=[(stats[i,0],components==i) for i in range(1,num) if stats[i,4]>=4]
        con.sort(key=lambda item:item[0])
        if path.stem in examples:
            with Image.open(path) as im: original=im.convert('RGB').resize((512,256),Image.Resampling.BILINEAR)
            canvas=Image.new('RGB',(512*5,290),'white');canvas.paste(original,(0,34));draw=ImageDraw.Draw(canvas);draw.text((5,7),'Original '+path.stem,fill='black')
        for idx,(name,left,right) in enumerate(BANDS,1):
            inside=np.zeros((256,256),bool);inside[:,left:right]=True
            outside_count=int(np.logical_and(line,~inside).sum())
            fractions=[float(np.logical_and(mask,~inside).sum()/mask.sum()) for _,mask in con]
            record=dict(image=path.stem,band=name,left=left,right=right,line_pixels=total,
                        line_pixels_outside=outside_count,line_fraction_hidden=(total-outside_count)/total if total else None,
                        connector_count=len(con),left_connector_visible=fractions[0] if len(con)==2 else None,
                        right_connector_visible=fractions[1] if len(con)==2 else None,
                        line_fully_hidden=bool(total and outside_count==0),
                        line_hidden_and_both_connectors_80=bool(total and outside_count==0 and len(con)==2 and min(fractions)>=.8))
            records.append(record)
            if path.stem in examples:
                panel=original.copy();d=ImageDraw.Draw(panel);d.rectangle((left*2,0,right*2-1,255),fill=(90,90,90))
                canvas.paste(panel,(idx*512,34));draw.text((idx*512+5,7),f'{name} ({left},{right})',fill='black')
        if path.stem in examples:canvas.save(OUT/'examples'/(path.stem+'_bands.png'))
    with (OUT/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    result={}
    for name,left,right in BANDS:
        rows=[r for r in records if r['band']==name]; good=[r for r in rows if r['line_fraction_hidden'] is not None]
        connectors=[r for r in rows if r['connector_count']==2]
        result[name]=dict(band_256_xyxy=[left,0,right,256],images=len(rows),line_present=len(good),
            line_fully_hidden=sum(r['line_fully_hidden'] for r in rows),mean_line_fraction_hidden=float(np.mean([r['line_fraction_hidden'] for r in good])) if good else None,
            exactly_two_connectors=len(connectors),mean_left_connector_visible=float(np.mean([r['left_connector_visible'] for r in connectors])) if connectors else None,
            mean_right_connector_visible=float(np.mean([r['right_connector_visible'] for r in connectors])) if connectors else None,
            hidden_and_both_connectors_80=sum(r['line_hidden_and_both_connectors_80'] for r in rows))
    summary=dict(protocol='Exploratory fixed full-height bands, chosen before audit of validation/good/010–029; existing automatically generated segmentation used only for diagnostics. No anomalous data, no training, no reference retrieval.',
                 line_fully_hidden_definition='all pixels of old segmentation classes 0–2 within band; not human ground truth',
                 visibility_80_definition='both class-3 connected-component masks retain >=80% area outside band; diagnostic threshold, not a validated socket-visibility criterion',
                 previously_inspected_note='010–019 were included in an earlier thumbnail overview; not a new untouched holdout',bands=result)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
