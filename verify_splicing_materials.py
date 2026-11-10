"""Verify table coverage and ten exported inputs/branch files."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace
from models.logico import LogiCo

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'saved_results/splicing_connectors_materials'
with (DEST/'逐图结果.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==312 and len({x['image_id'] for x in rows})==312
assert sum(x['image_type']=='good' for x in rows)==119
assert all((not r['spro_A'] and not r['spro_B']) if r['image_type']=='good' else (r['spro_A']!='' and r['spro_B']!='') for r in rows)
cases=json.loads((DEST/'代表样本清单.json').read_text(encoding='utf-8'))
assert len(cases)==10
for case in cases:
    folder=DEST/case['folder'];kind,name=case['image_id'].split('/')
    indices=np.asarray(Image.open(folder/'SMD输入_类别索引_256.png'))
    with np.load(folder/'SMD输入_onehot_256.npz') as f:onehot=f['onehot']
    assert onehot.shape==(5,256,256) and onehot.dtype==np.float32
    assert np.array_equal(onehot.argmax(axis=0),indices) and np.all(onehot.sum(axis=0)==1)
    with Image.open(folder/'组件分割_原始文件.png') as img:
        original512=torch.from_numpy(np.array(img.resize((512,512),Image.Resampling.NEAREST))).float().unsqueeze(0)
    resized=LogiCo.resize_seg_map(None,original512,size=256)
    actual=LogiCo.get_one_hot_encoding(SimpleNamespace(comp_num=5),resized)[0].numpy()
    assert np.array_equal(actual,onehot)
    original=ROOT/'saved_results/ab_splicing_connectors/maps'/kind/(Path(name).stem+'.npz')
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    assert sha(original)==sha(folder/'原始分支与融合图.npz')
    with np.load(original) as f:assert set(f.files)=={'CRN','SRN','SMD','A','B'}
    if kind!='good':
        assert len(list((folder/'GT_原始标注').glob('*.png')))==case['record']['annotation_count']
print('Verified 312 unique rows, normal blank sPRO, 10 one-hot inputs, exact branch file copies, and GT counts')
