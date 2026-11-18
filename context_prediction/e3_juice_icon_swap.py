"""Top fruit icon counterfactual. Fixed centre four tiles, normal images only."""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from context_prediction.e3_evaluate import ROOT, CACHE, tile_distances, best_ids
from context_prediction.tile_isolation import WEIGHT, encode_tiles
from dinov3.hub.backbones import load_dinov3_model

CAT = "juice_bottle"
DATA = ROOT/"data/mvtec_loco"/CAT
SEG = ROOT/"data/mvtec_loco_segmentations"/CAT
OUT = ROOT/"saved_results/context_e3_icon_swap"
VISIBLE = [i for i in range(16) if i not in (5, 6, 9, 10)]


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def inspect(path):
    with Image.open(SEG/path.relative_to(DATA)) as im:
        label = np.asarray(im)
    _, _, stats, _ = cv2.connectedComponentsWithStats((label == 3).astype("uint8"))
    h, w = label.shape
    x,y,bw,bh,_ = max((r for r in stats[1:] if .3*h < r[1] < .7*h),key=lambda r:r[4])
    with Image.open(path) as im:
        W,H = im.size
        # Replace only the printed interior of the upper sticker (including its yellow backing).
        box = tuple(round(v) for v in ((x+.15*bw)*W/w,(y+.14*bh)*H/h,(x+.85*bw)*W/w,(y+.86*bh)*H/h))
        cx = (x+.5*bw)*W/w
        patch = im.convert("RGB").crop((round(cx-.035*W),round((y/h-.095)*H),round(cx+.035*W),round((y/h-.055)*H)))
        color = np.median(np.asarray(patch).reshape(-1,3),axis=0)
    return box,color


@lru_cache(None)
def expected(ids):
    votes = []
    for name in ids:
        with np.load(CACHE/CAT/"train"/(name+".npz")) as z:
            centres = z["region_centers"]
            inside = ((centres >= .25) & (centres < .75)).all(axis=1)
            masks = z["masks"]
            vote = np.any(masks[inside],axis=0)
            h,w = vote.shape
            votes.append(vote[h//4:3*h//4,w//4:3*w//4])
    return np.stack(votes).sum(axis=0,dtype=np.uint8)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train = sorted((DATA/"train/good").glob("*.png"))
    val = sorted((DATA/"validation/good").glob("*.png"))
    donors, queries = val[:len(val)//2], val[len(val)//2:]
    props = {p:inspect(p) for p in train+val}
    colors = np.stack([props[p][1] for p in train])
    clusters = KMeans(n_clusters=3,random_state=1,n_init=10).fit(colors)
    # Semantic names for reporting only, matched to normal RGB colour prototypes.
    proto = np.array([[205,191,152],[204,156,60],[90,51,50]])
    ri,ci = linear_sum_assignment(np.linalg.norm(clusters.cluster_centers_[:,None]-proto,axis=2))
    types = {i:["banana","orange","cherry"][j] for i,j in zip(ri,ci)}
    def liquid(p):
        return types[np.argmin(np.linalg.norm(props[p][1]-clusters.cluster_centers_,axis=1))]
    assert digest(WEIGHT)==json.loads((CACHE/"config.json").read_text())["dino_sha256"]
    feats,labs=[],[]
    for p in train:
        with np.load(CACHE/CAT/"train"/(p.stem+".npz")) as z:
            assert str(z["source_sha256"])==digest(p)
            feats.append(z["tile_features"]);labs.append(z["tile_lab"])
    feats,labs=np.stack(feats),np.stack(labs)
    names=np.array([p.stem for p in train])
    ranked=np.argsort(np.linalg.norm((feats-feats.mean(axis=0)).reshape(len(train),-1),axis=1),kind="stable")
    fixed=tuple(names[ranked[np.linspace(0,len(train)-1,5,dtype=int)]])
    refs={p.stem:liquid(p) for p in train}
    model=load_dinov3_model("dinov3_vitb16",pretrained_weight_path=str(WEIGHT)).cuda().eval()
    records=[];montage=Image.new("RGB",(6*150,((len(queries)+5)//6)*96),"white");draw=ImageDraw.Draw(montage)
    for i,p in enumerate(queries):
        options=[d for d in donors if liquid(d)!=liquid(p)]
        d=options[i%len(options)]
        with Image.open(p) as im:source=im.convert("RGB")
        with Image.open(d) as im:icon=im.convert("RGB").crop(props[d][0])
        x1,y1,x2,y2=props[p][0];w,h=source.size
        assert w//4<=x1<x2<=3*w//4 and h//4<=y1<y2<=3*h//4
        swapped=source.copy();swapped.paste(icon.resize((x2-x1,y2-y1),Image.Resampling.BILINEAR),(x1,y1))
        delta=np.any(np.asarray(source)!=np.asarray(swapped),axis=2);changed=int(delta.sum())
        assert changed and changed==delta[y1:y2,x1:x2].sum()
        def compute(img):
            f,l=encode_tiles(img,model,"cuda:0")
            dist=tile_distances(dict(tile_features=f,tile_lab=l),feats,labs)
            return f,l,dict(fixed=fixed,full=tuple(best_ids(dist,names,list(range(16)))),masked=tuple(best_ids(dist,names,VISIBLE)))
        f,l,before=compute(source);g,m,after=compute(swapped)
        visible_diff=float(np.abs(f[VISIBLE]-g[VISIBLE]).max())
        assert visible_diff==0 and np.array_equal(l[VISIBLE],m[VISIBLE])
        left=(i%6)*150;top=(i//6)*96
        for j,img in enumerate((source,swapped)):
            montage.paste(img.crop((x1,y1,x2,y2)).resize((72,69)),(left+j*74,top+20))
        draw.text((left+2,top+2),p.stem+" "+liquid(p)+">"+liquid(d),fill="black")
        for mode in before:
            a,b=before[mode],after[mode]
            same_mask=np.array_equal(expected(a),expected(b))
            if mode=="masked": assert a==b and same_mask
            records.append(dict(image=p.stem,donor=d.stem,source_liquid=liquid(p),donor_liquid=liquid(d),mode=mode,
                ids_before=";".join(a),ids_after=";".join(b),same_ids=a==b,same_expected_mask=same_mask,
                matches_before=sum(refs[n]==liquid(p) for n in a),matches_after=sum(refs[n]==liquid(p) for n in b),
                donor_matches_after=sum(refs[n]==liquid(d) for n in b),box=props[p][0],changed_pixels=changed,visible_feature_difference=visible_diff))
    montage.save(OUT/"icon_pairs.jpg",quality=94)
    with (OUT/"queries.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    summary={}
    for mode in ("fixed","full","masked"):
        rs=[r for r in records if r["mode"]==mode]
        summary[mode]=dict(n=len(rs),same_ids=sum(r["same_ids"] for r in rs),same_mask=sum(r["same_expected_mask"] for r in rs),
            original_matches=sum(r["matches_before"] for r in rs),after_matches=sum(r["matches_after"] for r in rs),donor_matches=sum(r["donor_matches_after"] for r in rs))
    result=dict(training=len(train),classes={p:sum(liquid(img)==p for img in queries) for p in types.values()},centres={types[i]:v.tolist() for i,v in enumerate(clusters.cluster_centers_)},results=summary)
    (OUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result)
if __name__=="__main__": main()
