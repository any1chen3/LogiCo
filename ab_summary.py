"""Write a concise auditable report from paired experiment artifacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def write_report(category):
    folder=ROOT/'saved_results'/('breakfast_ab' if category=='breakfast_box' else 'ab_'+category)
    inference=json.loads((folder/'inference.json').read_text(encoding='utf-8'))
    spro=json.loads((folder/'spro_results.json').read_text(encoding='utf-8'))
    diag=json.loads((folder/'diagnosis/diagnosis.json').read_text(encoding='utf-8'))
    lines=[f'# {category} A/B 对照','',f"同一 checkpoint、测试 batch size=2、同一次前向推理；A 使用 CRN+SRN+SMD，B 不含 SMD。两组共用 A 的图像级分数，逻辑 iAUROC {inference['image_metrics']['Log-iAUC']*100:.4f}%。",'', '| 指标 | A | B | B−A（百分点） |','|---|---:|---:|---:|']
    a,b=spro['variants']['A'],spro['variants']['B']
    for key,label in [('logical_au_spro','逻辑 AU-sPRO'),('structural_au_spro','结构 AU-sPRO'),('mean_au_spro','逻辑与结构平均 AU-sPRO')]:
        lines.append(f"| {label} | {100*a[key]:.2f}% | {100*b[key]:.2f}% | {100*(b[key]-a[key]):+.2f} |")
    for name,part in a['subsets'].items():
        other=b['subsets'][name];x=part['au_spro'];y=other['au_spro']
        if x is None or y is None:lines.append(f"| {name} ({part['images']} 张) | 不适用 | 不适用 | 不适用 |")
        else:lines.append(f"| {name} ({part['images']} 张) | {100*x:.2f}% | {100*y:.2f}% | {100*(y-x):+.2f} |")
    lines.extend(['','子集按数据集原始缺陷名称定义，具体名称与每类数量见 spro_results.json。子集仅过滤缺陷通道，FPR 使用该方法下完整逻辑异常加正常图的背景；不是独立子集 FPR。小样本结果仅供探索。','','| 目标 FPR | A 实际 FPR | A 阈值 | A sPRO | B 实际 FPR | B 阈值 | B sPRO |','|---|---:|---:|---:|---:|---:|---:|'])
    for k in ('0.01','0.03','0.05'):
        x,y=diag['variants']['A']['points'][k],diag['variants']['B']['points'][k]
        lines.append(f"| {100*float(k):.0f}% | {100*x['actual_fpr']:.5f}% | {x['threshold']:.6f} | {100*x['spro']:.2f}% | {100*y['actual_fpr']:.5f}% | {y['threshold']:.6f} | {100*y['spro']:.2f}% |")
    lines.extend(['','阈值与工作点来自测试集，只用于探索，不是部署阈值。工作点经逐像素直接统计独立核对；曲线按 201 个目标 FPR 和背景直方图候选阈值构建。', '', '排名依据 1% 工作点的每区域 sPRO_B − sPRO_A，同一图片多区域取均值。五图分别显示原图、全部 GT 通道并集、A/B 二值图及 SMD 原始 sigmoid 热图；若某一方不足 5 张严格改善样本则只展示实际数量。','', '原模型 checkpoint 用测试集检测指标选取；数值是一次训练的探索性结果。保存的 sPRO 是饱和覆盖率，不等同于普通像素召回率。'])
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report saved:',folder/'report.md',flush=True)
