"""Exploratory E4 comparison using identical saved references and costs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
from context_prediction.e4_two_image_precheck import ROOT, WEIGHT, component_set, in_box, match_same_class, missing_vote
from context_prediction.evidence import component_cost
from dinov3.hub.backbones import load_dinov3_model

OUT=ROOT/'saved_results/context_e4_window_intersection'
SPECS=[
 ('connector_missing_026','splicing_connectors','test/logical_anomalies/026.png',range(4),ROOT/'saved_results/context_e4_two_image_precheck_display_fix/manifest.json','test_missing_026',ROOT/'saved_results/context_e4_two_image_precheck_display_fix/test_missing_026.npz'),
 ('connector_good_000','splicing_connectors','validation/good/000.png',range(4),ROOT/'saved_results/context_e4_two_image_precheck_display_fix/manifest.json','validation_good_000',ROOT/'saved_results/context_e4_two_image_precheck_display_fix/validation_good_000.npz'),
 ('breakfast_064','breakfast_box','test/logical_anomalies/064.png',range(6),ROOT/'saved_results/context_e4_breakfast_064/manifest.json',None,ROOT/'saved_results/context_e4_breakfast_064/breakfast_064.npz')]


def intersecting(items,clip):
    return [item for item in items if np.any(item['mask'] & clip)]


def union(items):
    out=np.zeros((256,256),bool)
    for item in items: out |= item['mask']
    return out


def image_panel(arr,size,kind='heat'):
    if kind=='mask': return Image.fromarray(np.uint8(arr)*255).resize(size,Image.Resampling.NEAREST).convert('RGB')
    colored=cv2.applyColorMap(np.uint8(np.clip(arr,0,1)*255),cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(cv2.resize(colored,size,interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2RGB))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    model=load_dinov3_model('dinov3_vitb16',pretrained_weight_path=str(WEIGHT)).cuda().eval()
    reference_cache={}; all_results={}
    for sample,category,relative,classes,manifest_file,sample_key,old_file in SPECS:
        source=ROOT/'data/mvtec_loco'/category/relative
        seg_path=ROOT/'data/mvtec_loco_segmentations'/category/relative
        original=json.loads(manifest_file.read_text(encoding='utf-8'))
        windows=original['samples'][sample_key]['windows'] if sample_key else original['windows']
        if len(windows)!=20 or any(len(w['reference_ids'])!=5 for w in windows): raise ValueError('Reference manifest incomplete')
        observed,seg=component_set(source,seg_path,model,class_ids=classes)
        raw_observed=union(observed)
        with np.load(old_file) as z: stored_e=z['evidence']
        maps={}; logs={}; window_arrays={}; selection_delta=[]
        for mode in ['centroid','intersects']:
            fused=np.zeros((256,256),np.float32); records=[]; inputs=[]; missing_full=[]; missing_clipped=[]; matched_inputs=[]
            for w in windows:
                x1,y1,x2,y2=[int(round(v*256)) for v in w['box_xyxy']]
                clip=np.zeros((256,256),bool);clip[y1:y2,x1:x2]=True
                actual=in_box(observed,w['box_xyxy']) if mode=='centroid' else intersecting(observed,clip)
                if mode=='intersects':
                    cent=in_box(observed,w['box_xyxy']); selection_delta.append((len(actual)-len(cent),w['window']))
                inputs.append(union(actual)); votes=[]; ref_logs=[]; per_full=[]; per_clip=[]; per_match=[]
                for ref in w['reference_ids']:
                    key=(category,ref)
                    if key not in reference_cache:
                        p=ref+'.png'
                        reference_cache[key]=component_set(ROOT/'data/mvtec_loco'/category/'train/good'/p,ROOT/'data/mvtec_loco_segmentations'/category/'train/good'/p,model,class_ids=classes)[0]
                    expected=in_box(reference_cache[key],w['box_xyxy']) if mode=='centroid' else intersecting(reference_cache[key],clip)
                    pairs,missing=match_same_class(expected,actual)
                    _,full_missing=missing_vote(expected,actual,pairs,np.ones((256,256),bool))
                    clipped=full_missing & clip
                    votes.append(clipped);per_full.append(full_missing);per_clip.append(clipped)
                    paired_actual=[actual[j] for j in sorted(set(pairs.values()))];per_match.append(union(paired_actual))
                    ref_logs.append({'reference_id':ref,'expected_ids':[x['id'] for x in expected],
                                     'observed_ids':[x['id'] for x in actual],
                                     'pairs':[{'expected':expected[i]['id'],'observed':actual[j]['id'],
                                               'class_id':expected[i]['class_id'],'cost':component_cost(expected[i],actual[j])} for i,j in sorted(pairs.items())],
                                     'unmatched_expected':[expected[i]['id'] for i in missing]})
                fused=np.maximum(fused,np.mean(votes,axis=0))
                missing_full.append(per_full);missing_clipped.append(per_clip);matched_inputs.append(per_match)
                records.append({'window':w['window'],'box_xyxy':w['box_xyxy'],'reference_ids':w['reference_ids'],
                                'observed_ids':[x['id'] for x in actual],'references':ref_logs})
            maps[mode]=fused;logs[mode]=records
            window_arrays[mode+'_observed_inputs']=np.asarray(inputs,bool)
            window_arrays[mode+'_matched_observed']=np.asarray(matched_inputs,bool)
            window_arrays[mode+'_missing_before_clip']=np.asarray(missing_full,bool)
            window_arrays[mode+'_missing_after_clip']=np.asarray(missing_clipped,bool)
        difference=float(np.max(np.abs(maps['centroid']-stored_e)))
        if difference>1e-6: raise RuntimeError(f'{sample}: original E did not reproduce: {difference}')
        sample_dir=OUT/sample;sample_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(sample_dir/'window_masks.npz',raw_observed=raw_observed,original_seg=seg,
                            evidence_centroid=maps['centroid'],evidence_intersects=maps['intersects'],**window_arrays)
        # Windows are chosen solely from component/window geometry, without GT.
        candidate=sorted(selection_delta,key=lambda pair:(-pair[0],pair[1]))[0][1]
        wi=next(i for i,w in enumerate(windows) if w['window']==candidate)
        box=windows[wi]['box_xyxy']; pixbox=tuple(round(x*256) for x in box)
        size=(512,256) if category=='splicing_connectors' else (512,410)
        with Image.open(source) as im: original_image=im.convert('RGB').resize(size,Image.Resampling.BILINEAR)
        matching_input=window_arrays['intersects_observed_inputs'][wi]
        remaining=np.mean(window_arrays['intersects_missing_after_clip'][wi],axis=0)
        panels=[(original_image,'Original'),(image_panel(raw_observed,size,'mask'),'Raw observed segmentation'),
                (image_panel(matching_input,size,'mask'),f'Matching input: {candidate} (full masks)'),
                (image_panel(remaining,size),f'Unmatched evidence: {candidate} (0-1)'),
                (image_panel(maps['centroid'],size),'Old E: centroid assignment (0-1)'),
                (image_panel(maps['intersects'],size),'New E: window intersection (0-1)')]
        width,height=size; canvas=Image.new('RGB',(width*3,(height+24)*2),'white'); draw=ImageDraw.Draw(canvas)
        for i,(img,label) in enumerate(panels):
            x=(i%3)*width;y=(i//3)*(height+24);canvas.paste(img,(x,y+24));draw.text((x+5,y+3),label,fill='black')
        canvas.save(sample_dir/'comparison.png')
        stats={'sample':sample,'category':category,'image':str(source),'observed_segmentation':str(seg_path),
               'selected_window':candidate,'selected_box_256':pixbox,'old_max_difference_from_saved':difference,
               'reference_ids_preserved':True,'components':[{'id':o['id'],'class_id':o['class_id'],'area':float(o['area'])} for o in observed],
               'windows':logs}
        (sample_dir/'matches.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
        category_mask=(seg==3) if sample.startswith('connector') else (seg==1)
        val={k:float(v[category_mask].mean()) if category_mask.any() else None for k,v in maps.items()}
        all_results[sample]=dict(selected_window=candidate,original_reproduced=difference,observed_components=len(observed),
                                 observed_class=3 if category=='splicing_connectors' else 1,
                                 mean_evidence_on_observed_class=val)
        print(sample,json.dumps(all_results[sample]),flush=True)
    (OUT/'summary.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
