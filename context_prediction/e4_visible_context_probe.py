"""Normal-only fixed/masked/full reference comparison; line signatures are evaluation-only."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from context_prediction.e3_evaluate import CACHE, ROOT, queries, read, tile_distances, best_ids
from context_prediction.e4_normal_alignment_probe import read_components,anchors,end_height
from context_prediction.e4_two_image_precheck import WEIGHT, visible_key
from context_prediction.tile_isolation import boxes
from dinov3.hub.backbones import load_dinov3_model

CAT='splicing_connectors'
DATA=ROOT/'data/mvtec_loco'/CAT
SEG=ROOT/'data/mvtec_loco_segmentations'/CAT
OUT=ROOT/'saved_results/context_e4_visible_context_probe'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def signature(path):
    components=read_components(path); a=anchors(components)
    lines=[x for x in components.values() if x['class_id'] in (0,1,2)]
    if a is None:return None,'not_exactly_two_connectors'
    if len(lines)!=1:return None,'not_exactly_one_line'
    heights=end_height(lines[0]['mask'],a)
    if heights is None:return None,'invalid_line_endpoints'
    return {'color':lines[0]['class_id'],'heights':heights,'mask':lines[0]['mask']},None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=sorted((CACHE/CAT/'train').glob('*.npz'))
    if len(train)!=360:raise ValueError('Training cache incomplete')
    config=json.loads((CACHE/'config.json').read_text(encoding='utf-8'))
    if config['dino_sha256']!=sha(WEIGHT):raise ValueError('Encoder differs from E3 cache')
    bank=[(p,read(p)) for p in train]; names=np.asarray([p.stem for p,_ in bank])
    features=np.stack([d['tile_features'] for _,d in bank]);labs=np.stack([d['tile_lab'] for _,d in bank])
    mean=np.mean(features,axis=0); fixed_rank=sorted((float(np.linalg.norm(d['tile_features']-mean)),p.stem) for p,d in bank)
    fixed=[fixed_rank[i][1] for i in np.linspace(0,len(fixed_rank)-1,5,dtype=int)]
    ref_signatures={}; invalid_ref={}
    for name in names:
        sig,reason=signature(SEG/'train/good'/(name+'.png'))
        if sig is None:invalid_ref[reason]=invalid_ref.get(reason,0)+1
        ref_signatures[name]=sig
    images=sorted((DATA/'validation/good').glob('*.png'))[:10]
    observed={};invalid_observed={}; rows=[];per_image=[]
    for path in images:
        sig,reason=signature(SEG/'validation/good'/path.name)
        if sig is None:invalid_observed[path.stem]=reason;continue
        query=read(CACHE/CAT/'validation'/(path.stem+'.npz'))
        if str(query['source_sha256'])!=sha(path):raise ValueError('Validation image differs from cache')
        observed[path.stem]=sig
        distances=tile_distances(query,features,labs)
        full=best_ids(distances,names,list(range(16)))
        group=[]
        for window,hidden,box in queries():
            x1,y1,x2,y2=[round(v*256) for v in box]; mask=sig['mask']
            if not mask[y1:y2,x1:x2].any():continue
            visible=[i for i in range(16) if i not in hidden]
            selected={'fixed':fixed,'full':full,'masked':best_ids(distances,names,visible)}
            yy,xx=np.nonzero(mask);left,right=np.quantile(xx,[.15,.85]);end_pixels=((xx<=left)|(xx>=right))
            endpoints_visible=bool(np.any(end_pixels & ~((xx>=x1)&(xx<x2)&(yy>=y1)&(yy<y2))))
            for method,ids in selected.items():
                usable=[ref_signatures[id] for id in ids if ref_signatures[id] is not None]
                same=[r for r in usable if r['color']==sig['color']]
                diffs=[float(np.mean(np.abs(np.asarray(r['heights'])-sig['heights']))) for r in same]
                r=dict(image=path.stem,window=window,method=method,reference_ids=';'.join(ids),
                       actual_color=sig['color'],visible_endpoint_pixels=endpoints_visible,
                       usable_references=len(usable),same_color_references=len(same),
                       mean_endpoint_error_same_color=float(np.mean(diffs)) if diffs else '',
                       endpoint_error_count=len(diffs))
                rows.append(r);group.append(r)
        for method in ('fixed','full','masked'):
            g=[r for r in group if r['method']==method]
            per_image.append(dict(image=path.stem,method=method,query_count=len(g),
                                  mean_usable_references=float(np.mean([r['usable_references'] for r in g])) if g else None,
                                  mean_same_color_count=float(np.mean([r['same_color_references'] for r in g])) if g else None,
                                  queries_with_color_match=sum(r['endpoint_error_count']>0 for r in g),
                                  mean_endpoint_error_given_color=float(np.mean([r['mean_endpoint_error_same_color'] for r in g if r['endpoint_error_count']])) if any(r['endpoint_error_count'] for r in g) else None))
    for filename,items in [('queries.csv',rows),('per_image.csv',per_image)]:
        if not items:raise ValueError('No eligible normal queries')
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=items[0]);w.writeheader();w.writerows(items)
    model=load_dinov3_model('dinov3_vitb16',pretrained_weight_path=str(WEIGHT)).cuda().eval()
    checks=[]; selected_windows={'small_0_0','small_1_1','small_2_2','small_3_3','large_0_0'}
    for idx in range(3):
        with Image.open(images[idx]) as im:source=im.convert('RGB')
        with Image.open(images[idx+1]) as im:donor=im.convert('RGB')
        if source.size!=donor.size:raise ValueError('Source and donor size mismatch')
        grid=boxes(source.width,source.height)
        for name,hidden,box in queries():
            if name not in selected_windows:continue
            swapped=source.copy()
            for i in hidden:swapped.paste(donor.crop(grid[i]),grid[i])
            changed=any(not np.array_equal(np.asarray(source.crop(grid[i])),np.asarray(donor.crop(grid[i]))) for i in hidden)
            visible=[i for i in range(16) if i not in hidden]
            a=visible_key(source,visible,model);b=visible_key(swapped,visible,model)
            before=best_ids(tile_distances(a,features,labs),names,visible)
            after=best_ids(tile_distances(b,features,labs),names,visible)
            hashes_before=[sha(SEG/'train/good'/(x+'.png')) for x in before]
            hashes_after=[sha(SEG/'train/good'/(x+'.png')) for x in after]
            checks.append(dict(source=images[idx].name,donor=images[idx+1].name,window=name,changed=changed,
                               visible_feature_max_diff=float(np.max(np.abs(a['tile_features'][visible]-b['tile_features'][visible]))),
                               visible_lab_max_diff=float(np.max(np.abs(a['tile_lab'][visible]-b['tile_lab'][visible]))),
                               reference_ids_equal=before==after,reference_seg_hashes_equal=hashes_before==hashes_after,
                               reference_ids=';'.join(before)))
    with (OUT/'target_replacement.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=checks[0]);w.writeheader();w.writerows(checks)
    summary={'scope':'first 10 sorted normal validation images; diagnostic line endpoints from original automatically generated segmentation are evaluation-only and never retrieval inputs',
             'training_references':len(train),'invalid_training_signatures':invalid_ref,'invalid_validation_signatures':invalid_observed,
             'eligible_queries':sum(r['method']=='masked' for r in rows),'fixed_reference_ids':fixed,'images':per_image,
             'target_replacement_checks':len(checks),'target_replacement_passes':sum(r['changed'] and r['visible_feature_max_diff']<=1e-6 and r['visible_lab_max_diff']==0 and r['reference_ids_equal'] and r['reference_seg_hashes_equal'] for r in checks)}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='images'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
