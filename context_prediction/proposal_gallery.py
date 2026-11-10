"""Build contact sheets for human review of E2 proposals; no labels inferred."""
import json
import html
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'saved_results/context_e2_proposals'
OUT = ROOT / 'saved_results/context_e2_gallery'


def main():
    count = 0
    pages = []
    for metadata_path in sorted(DATA.glob('*/*/metadata.json')):
        info = json.loads(metadata_path.read_text(encoding='utf-8'))
        with Image.open(info['image']) as image:
            base = image.convert('RGB').resize((160, 160), Image.Resampling.BILINEAR)
        with np.load(metadata_path.parent / 'masks.npz') as data:
            masks = data['masks']
        target = OUT / info['category'] / Path(info['image']).stem
        target.mkdir(parents=True, exist_ok=True)
        base.save(target / 'original.png')
        for start in range(0, len(masks), 16):
            sheet = Image.new('RGB', (640, 720), 'white')
            for offset, mask in enumerate(masks[start:start+16]):
                alpha = Image.fromarray(mask.astype(np.uint8) * 110).resize((160, 160), Image.Resampling.NEAREST)
                overlay = base.copy()
                overlay.paste((255, 35, 35), (0, 0, 160, 160), alpha)
                x, y = offset % 4 * 160, offset // 4 * 180
                sheet.paste(overlay, (x, y))
                ImageDraw.Draw(sheet).text((x+3, y+161), f'candidate {start+offset:03d}', fill='black')
            sheet.save(target / f'candidates_{start:03d}_{min(start+15, len(masks)-1):03d}.jpg', quality=85)
        links = sorted(target.glob('candidates_*.jpg'))
        category = html.escape(info['category'])
        name = html.escape(Path(info['image']).stem)
        thumbnails = ''.join(f'<a href="{html.escape(p.relative_to(OUT).as_posix())}">{html.escape(p.stem)}</a> ' for p in links)
        pages.append(f'<section><h3>{category} / {name}（{len(masks)} 个候选）</h3>{thumbnails}</section>')
        count += 1
    (OUT / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>组件候选审阅</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto;padding:25px}a{display:inline-block;margin:5px}section{border-bottom:1px solid #ddd}</style><h1>组件候选审阅</h1><p>每张联系表将红色区域叠加在原图上；候选编号与对应 metadata.json 一致。候选数量不代表组件覆盖率。缩略图统一缩放为正方形，仅用于浏览。</p>'+''.join(pages), encoding='utf-8')
    print(f'Created contact sheets for {count} images in {OUT}')


if __name__ == '__main__':
    main()
