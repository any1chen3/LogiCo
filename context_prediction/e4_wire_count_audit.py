"""Audit existing component segmentation wire counts; no model inference."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from context_prediction.e4_normal_alignment_probe import read_components

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/mvtec_loco/splicing_connectors'
SEG=ROOT/'data/mvtec_loco_segmentations/splicing_connectors'
LABELS=ROOT/'saved_results/ab_splicing_connectors/diagnosis/region_differences.csv'
OUT=ROOT/'saved_results/context_e4_wire_count_audit'


def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'examples').mkdir(exist_ok=True)
    defects=defaultdict(set)
    with LABELS.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f): defects[r['image']].add(r['defect_name'])
    selected={name for name,names in defects.items() if 'missing_cable' in names}
    if len(selected)!=12:raise ValueError(f'Expected 12 missing_cable annotations, got {len(selected)}')
    rows=[]
    for split in ('train/good','validation/good','test/good','test/logical_anomalies'):
        paths=sorted((SEG/split).glob('*.png'))
        if not paths: raise ValueError('No segmentation masks in '+split)
        for p in paths:
            original=DATA/split/p.name
            if not original.exists(): raise ValueError('Image missing: '+str(original))
            components=read_components(p)
            lines=[v for v in components.values() if v['class_id'] in (0,1,2)]
            con=[v for v in components.values() if v['class_id']==3]
            rows.append(dict(split=split,image=p.name,defects=';'.join(sorted(defects[p.name])) if split=='test/logical_anomalies' else '',
                missing_cable_annotated=bool(split=='test/logical_anomalies' and p.name in selected),
                line_components=len(lines),line_class_counts=';'.join(str(sum(v['class_id']==i for v in lines)) for i in range(3)),
                line_pixels=sum(v['area'] for v in lines),connector_components=len(con),
                no_wire_detected=len(lines)==0,connector_count_two=len(con)==2))
    with (OUT/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary_splits={}
    for split in ('train/good','validation/good','test/good','test/logical_anomalies'):
        rs=[r for r in rows if r['split']==split]
        summary_splits[split]=dict(images=len(rs),line_count_distribution=dict(sorted(Counter(str(r['line_components']) for r in rs).items())),
            no_wire_detected=sum(r['no_wire_detected'] for r in rs),connector_count_two=sum(r['connector_count_two'] for r in rs))
    missing=[r for r in rows if r['missing_cable_annotated']]
    rest=[r for r in rows if r['split']=='test/logical_anomalies' and not r['missing_cable_annotated']]
    other_no=[dict(image=r['image'],defects=r['defects']) for r in rest if r['no_wire_detected']]
    summary=dict(protocol='Read previously generated LogiCo component label maps, resize by nearest neighbor to 256x256; count connected components >=4 pixels for labels 0,1,2. Defect names from existing test annotation export, used only for retrospective grouping. No retraining, new segmentation, thresholds fitted on test, checkpoint or inference.',
        rule='Normal training counts are all 1; flag missing candidate when no wire-class component is detected. This flags absence of segmentation, not necessarily physical absence or a specific anomaly cause.',
        splits=summary_splits,missing_cable=dict(images=len(missing),no_wire_detected=sum(r['no_wire_detected'] for r in missing)),
        remaining_logical=dict(images=len(rest),no_wire_detected=len(other_no),breakdown=dict(Counter(x['defects'] for x in other_no)),examples=other_no),
        caveats='Saved segmentations only; false negatives/positive components may be segmentation errors and splitting/merging can change counts; no independent physical count labels; no sPRO, calibrated probability, real-time detector or component-location evaluation.')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    examples=[('test/logical_anomalies','026'),('test/logical_anomalies','003'),('test/logical_anomalies','059'),('validation/good','013')]
    for split,stem in examples:
        with Image.open(DATA/split/(stem+'.png')) as im: original=im.convert('RGB').resize((512,256))
        with Image.open(SEG/split/(stem+'.png')) as im: lab=np.asarray(im.resize((256,256),Image.Resampling.NEAREST))
        wire=np.isin(lab,[0,1,2]); connector=lab==3
        overlay=np.stack((wire*255,connector*220,np.zeros_like(lab)),axis=-1).astype('uint8')
        overlay=Image.fromarray(overlay).resize((512,256),Image.Resampling.NEAREST)
        panel=Image.new('RGB',(1024,286),'white'); panel.paste(original,(0,30));panel.paste(overlay,(512,30));d=ImageDraw.Draw(panel)
        d.text((5,7),split+' / '+stem,fill='black');d.text((517,7),'Existing segmentation: red wire, green connector',fill='black')
        panel.save(OUT/'examples'/(split.replace('/','_')+'_'+stem+'.jpg'),quality=90)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
