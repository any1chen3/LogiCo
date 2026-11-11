"""Exact mask-IoU deduplication using bit-packed binary masks."""
import numpy as np

_BITS = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def select_masks_fast(proposals, height, width):
    filtered=[]
    for index,p in enumerate(proposals):
        mask=p['segmentation']
        pixels=int(mask.sum()); area=pixels*(512*512)/(height*width)
        if 16 <= area <= .8*512*512:
            x,y,w,h=p['bbox']
            filtered.append((p,index,area,(int(x),int(y),int(w),int(h)),pixels))
    filtered.sort(key=lambda v:(-float(v[0]['predicted_iou']),-float(v[0]['stability_score']),v[3],v[1]))
    if not filtered: return []
    packed=np.stack([np.packbits(v[0]['segmentation'].reshape(-1)) for v in filtered])
    selected=[]
    for i,item in enumerate(filtered):
        if selected:
            intersections=_BITS[np.bitwise_and(packed[selected],packed[i])].sum(axis=1,dtype=np.int64)
            unions=np.asarray([filtered[j][4] for j in selected],dtype=np.int64)+item[4]-intersections
            if np.any(intersections / unions > .85): continue
        selected.append(i)
    return [filtered[i][:4] for i in selected]
