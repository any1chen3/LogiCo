"""Visualize one cached masked-reference retrieval example per category."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from context_prediction.e3_evaluate import CACHE, OUT, queries


def main():
    with (OUT/'masked_prediction.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    query_boxes={name:box for name,_,box in queries()}
    for category in ('splicing_connectors','juice_bottle'):
        row=next(r for r in rows if r['category']==category and r['mode']=='masked' and r['query']=='large_0_0' and int(r['observed_count'])>0)
        reference=row['reference_ids'].split(';')[0]
        with np.load(CACHE/category/'validation'/(row['image']+'.npz')) as f:
            source_path=str(f['source_path'])
        with np.load(CACHE/category/'train'/reference) as f:
            refpath=str(f['source_path']); masks=f['masks']; centers=f['region_centers']
        with Image.open(source_path) as im: source=im.convert('RGB').resize((384,384))
        with Image.open(refpath) as im: ref=im.convert('RGB').resize((384,384))
        x1,y1,x2,y2=query_boxes[row['query']]
        box=(round(x1*384),round(y1*384),round(x2*384),round(y2*384))
        masked=source.copy(); ImageDraw.Draw(masked).rectangle((box[0],box[1],box[2]-1,box[3]-1),fill=(90,90,90))
        ids=np.flatnonzero((centers[:,0]>=x1)&(centers[:,0]<x2)&(centers[:,1]>=y1)&(centers[:,1]<y2))
        union=np.any(masks[ids],axis=0) if len(ids) else np.zeros(masks.shape[1:],bool)
        m=Image.fromarray(union.astype(np.uint8)*120).resize((384,384),Image.Resampling.NEAREST)
        clip=np.zeros((384,384),dtype=np.uint8); clip[box[1]:box[3],box[0]:box[2]]=np.asarray(m)[box[1]:box[3],box[0]:box[2]]
        over=ref.copy(); over.paste((40,130,255),(0,0,384,384),Image.fromarray(clip))
        canvas=Image.new('RGB',(384*4,424),'white')
        for i,img in enumerate((source,masked,ref,over)): canvas.paste(img,(384*i,40))
        ImageDraw.Draw(canvas).text((5,5),f"{category}: source {row['image']} | masked target | reference {reference} | expected union (first reference only)",fill='black')
        dest=OUT/'examples'; dest.mkdir(exist_ok=True); canvas.save(dest/(category+'.jpg'),quality=90)
        print(category,row['image'],reference,len(ids))


if __name__=='__main__': main()
