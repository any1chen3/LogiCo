"""Exact threshold counts via sorted pixels; sPRO thresholds remain unchanged."""
import numpy as np
import utils.spro.metrics as metrics

ORIGINAL_TP=metrics.get_tp_areas_for_thresholds
ORIGINAL_FP=metrics.get_fp_areas_for_thresholds
ORIGINAL_TN=metrics.get_tn_areas_for_thresholds
_background={}
_regions={}

def clear():
    _background.clear(); _regions.clear()

def tp(gt_channel, anomaly_map, anomaly_thresholds):
    key=(id(gt_channel),id(anomaly_map))
    if key not in _regions:
        _regions[key]=np.sort(anomaly_map.np_array[gt_channel.bool_array])
    values=_regions[key]
    return len(values)-np.searchsorted(values,np.asarray(anomaly_thresholds,dtype=np.float64),side='right')

def fp(gt_map, anomaly_map, anomaly_thresholds):
    key=id(anomaly_map)
    if key not in _background:
        values=anomaly_map.np_array.ravel() if gt_map is None else anomaly_map.np_array[~gt_map.get_or_over_channels()]
        _background[key]=np.sort(values)
    values=_background[key]
    return len(values)-np.searchsorted(values,np.asarray(anomaly_thresholds,dtype=np.float64),side='right')

def tn(gt_map, anomaly_map, anomaly_thresholds, fp_areas=None):
    if fp_areas is None: fp_areas=fp(gt_map,anomaly_map,anomaly_thresholds)
    return len(_background[id(anomaly_map)])-fp_areas

def validate_and_install(gt_maps, anomaly_maps):
    """Compare exact counts, including thresholds equal to a pixel value."""
    selected=[]
    for want_gt in (False,True):
        chosen=next(((g,a) for g,a in zip(gt_maps,anomaly_maps) if (g is not None)==want_gt),None)
        if chosen: selected.append(chosen)
    for gt,amap in selected:
        flat=amap.np_array.ravel()
        thresholds=np.array([float(flat[0]),float(np.percentile(flat,50)),float(np.percentile(flat,99)),float(flat.max())],dtype=np.float64)
        old_fp=ORIGINAL_FP(gt,amap,thresholds)
        new_fp=fp(gt,amap,thresholds)
        assert np.array_equal(old_fp,new_fp), 'accelerated FP mismatch'
        assert np.array_equal(ORIGINAL_TN(gt,amap,thresholds,old_fp),tn(gt,amap,thresholds,new_fp)), 'accelerated TN mismatch'
        if gt:
            for channel in gt.channels:
                assert np.array_equal(ORIGINAL_TP(channel,amap,thresholds),tp(channel,amap,thresholds)), 'accelerated TP mismatch'
    clear()
    metrics.get_tp_areas_for_thresholds=tp
    metrics.get_fp_areas_for_thresholds=fp
    metrics.get_tn_areas_for_thresholds=tn
