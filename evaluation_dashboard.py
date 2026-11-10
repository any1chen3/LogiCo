"""Local read-only dashboard for the four LogiCo evaluations."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import csv
import re
import subprocess
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'saved_results' / 'four_class_evaluation'
CATS = ('breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors')
PAGE = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LogiCo 测试进度</title><style>
:root{font-family:system-ui,'Microsoft YaHei',sans-serif;color:#17263b;background:#f4f7fb}body{max-width:1000px;margin:0 auto;padding:36px 22px}h1{font-size:30px;margin:8px 0}.sub{color:#61738c;line-height:1.6}.summary,.panel,.card{background:white;border:1px solid #e5ebf3;box-shadow:0 8px 28px #2436570a;border-radius:16px;padding:20px}.summary{margin:24px 0;display:flex;justify-content:space-between;gap:18px;align-items:center}.summary strong{font-size:24px;color:#174b94}progress{width:100%;height:14px;accent-color:#2563eb}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}.card h2{font-size:17px;margin:0 0 12px}.badge{font-size:13px;color:#435a76}.metrics{line-height:1.9;margin-top:14px;font-size:14px}.panel{margin-top:18px}pre{white-space:pre-wrap;word-break:break-all;color:#495d77;font-size:13px}img{max-width:100%;border-radius:8px}.note{font-size:13px;color:#71829a}
</style></head><body><span class="sub">LOGICO / MVTec LOCO</span><h1>五类模型测试</h1><p class="sub">每 5 秒更新。指标来自项目原版图像级评估。</p><div class="summary"><div><span class="sub">已完成测试</span><br><strong id="count">—</strong></div><div style="flex:1"><p id="current">读取中…</p><progress id="bar" max="5" value="0"></progress></div></div><div class="grid" id="cards"></div><div class="panel"><h2>测试样例</h2><p class="note">原图 / 异常热图 / 标注。选择标注仅切换显示，不会重新计算指标。</p><label>类别 <select id="select"></select></label> <label>图片 <select id="imageSelect"></select></label> <label>标注 <select id="annotationSelect"></select></label><p id="previewmsg">等待生成可视化图片</p><img id="preview" alt="测试图像、异常热图和标注"></div><div class="panel"><h2>测试进程</h2><pre id="log"></pre><p class="note">原训练程序使用测试集指标选择最佳模型，结果可能偏乐观。定位 sPRO 未包含在本次原版测试中。</p></div><script>
let selected='', selectedImage='', selectedAnnotation='', loadedCategory='';const label={breakfast_box:'早餐盒',juice_bottle:'果汁瓶',pushpins:'图钉',screw_bag:'螺丝袋',splicing_connectors:'拼接连接器'};
let imageInfo=[];async function loadImages(category){let r=await fetch('/api/images?category='+encodeURIComponent(category),{cache:'no-store'}), images=await r.json();imageInfo=images;let s=document.getElementById('imageSelect');s.replaceChildren();for(const item of images){let option=document.createElement('option');option.value=item.index;option.textContent=item.name;s.append(option)}selectedImage=images.length?String(images[0].index):'';loadedCategory=category;await loadAnnotations()}
async function loadAnnotations(){let s=document.getElementById('annotationSelect');s.replaceChildren();if(!selected || selectedImage===''){selectedAnnotation='';showImage();return}let url='/api/annotations?category='+encodeURIComponent(selected)+'&index='+encodeURIComponent(selectedImage);let list=await(await fetch(url,{cache:'no-store'})).json();for(const name of list){let option=document.createElement('option');option.value=name;option.textContent=name;s.append(option)}if(!list.length){let option=document.createElement('option');option.value='';option.textContent='无标注（正常图片）';s.append(option)}selectedAnnotation=s.value;showImage()}
function showImage(){let im=document.getElementById('preview');if(selected && selectedImage!==''){im.src='/api/preview?category='+encodeURIComponent(selected)+'&index='+encodeURIComponent(selectedImage)+'&annotation='+encodeURIComponent(selectedAnnotation);document.getElementById('previewmsg').textContent=''}else{im.removeAttribute('src');document.getElementById('previewmsg').textContent='该类别暂无可视化图片'}updatePrediction()}
function updatePrediction(){let item=imageInfo.find(x=>String(x.index)===selectedImage);let el=document.getElementById('prediction');if(!item){el.textContent='';return}let score=item.score;el.textContent=score===null || score===undefined?'图像异常分数正在准备中':'模型图像级异常分数：'+Number(score).toFixed(5);}
async function update(){try{let d=await(await fetch('/api/status',{cache:'no-store'})).json();document.getElementById('count').textContent=d.finished+' / 5';document.getElementById('bar').value=d.finished;document.getElementById('current').textContent=d.current?'当前：'+(label[d.current]||d.current)+' · '+d.progress:d.state;document.getElementById('cards').innerHTML=d.items.map(x=>`<div class="card"><h2>${label[x.name]} <small class="badge">${x.name}</small></h2><span class="badge">${x.status}</span><div class="metrics">${x.metrics?`逻辑异常 iAUC：${x.metrics[0]}%<br>结构异常 iAUC：${x.metrics[1]}%<br>平均 iAUC：${x.metrics[2]}%`:'等待结果'}</div></div>`).join('');document.getElementById('log').textContent=d.log;let ready=d.items.filter(x=>x.preview);let s=document.getElementById('select');let old=selected||s.value;s.replaceChildren();for(const x of ready){let option=document.createElement('option');option.value=x.name;option.textContent=label[x.name];s.append(option)}if(ready.some(x=>x.name===old))s.value=old;selected=s.value;if(selected && selected!==loadedCategory)await loadImages(selected);else if(selected && imageInfo.some(x=>x.score===null)){let fresh=await(await fetch('/api/images?category='+encodeURIComponent(selected),{cache:'no-store'})).json();imageInfo=fresh;updatePrediction()}}catch(e){document.getElementById('current').textContent='连接中断，请检查本地服务'}}document.getElementById('select').onchange=async e=>{selected=e.target.value;await loadImages(selected)};document.getElementById('imageSelect').onchange=async e=>{selectedImage=e.target.value;await loadAnnotations()};document.getElementById('annotationSelect').onchange=e=>{selectedAnnotation=e.target.value;showImage()};document.getElementById('threshold').oninput=e=>{localStorage.setItem('logico_threshold_'+selected,e.target.value);updatePrediction()};update();setInterval(update,5000)
</script></body></html>'''

PAGE = PAGE.replace('<p id="previewmsg">', '<p id="prediction"></p><p class="note">分数越高表示模型认为异常程度越高；这里不设置正常或异常的判断阈值。</p><p id="previewmsg">')
PAGE = PAGE.replace("document.getElementById('threshold').oninput=e=>{localStorage.setItem('logico_threshold_'+selected,e.target.value);updatePrediction()};", '')
PAGE = PAGE.replace('原图 / 异常热图 / 标注。选择标注仅切换显示，不会重新计算指标。', '原图 → GT → A 热力图 → B 热力图 → SMD 热力图 → A、B 二值图。切换标注仅改变 GT 显示，不重新计算指标。')
PAGE = PAGE.replace('<img id="preview" alt="测试图像、异常热图和标注">', '''<p class="note">A/B 热图按当前图片共同色标显示；SMD 显示原始 sigmoid（0–1）。二值图使用对应类别诊断的约 1% FPR 工作点，阈值取自测试集，仅供诊断；尚未计算的数据会标明。</p><div class="panel-grid" id="panels"></div>''')
PAGE = PAGE.replace('</style>', '.panel-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.tile{min-width:0}.tile h3{font-size:14px;margin:8px 0}.tile img{width:100%;display:block}.placeholder{height:170px;display:grid;place-items:center;background:#f0f3f7;color:#687b93;border-radius:8px;font-size:13px}</style>')
PAGE = PAGE.replace("function showImage(){let im=document.getElementById('preview');if(selected && selectedImage!==''){im.src='/api/preview?category='+encodeURIComponent(selected)+'&index='+encodeURIComponent(selectedImage)+'&annotation='+encodeURIComponent(selectedAnnotation);document.getElementById('previewmsg').textContent=''}else{im.removeAttribute('src');document.getElementById('previewmsg').textContent='该类别暂无可视化图片'}updatePrediction()}", """function showImage(){let box=document.getElementById('panels');box.replaceChildren();let item=imageInfo.find(x=>String(x.index)===selectedImage);let parts=[['image','原图'],['gt','GT'],['A','A 热力图'],['B','B 热力图'],['SMD','SMD 热力图'],['A_binary','A 二值图'],['B_binary','B 二值图']];if(selected && selectedImage!=='' && item){for(let [part,title] of parts){let tile=document.createElement('div');tile.className='tile';let h=document.createElement('h3');h.textContent=title;tile.append(h);let unavailable=(!item.has_ab && ['B','SMD','A_binary','B_binary'].includes(part))||(!item.has_binary && ['A_binary','B_binary'].includes(part));if(unavailable){let missing=document.createElement('div');missing.className='placeholder';missing.textContent=item.has_ab?'诊断阈值计算中':'尚无 A/B 数据';tile.append(missing)}else{let im=document.createElement('img');im.alt=title;im.src='/api/panel?category='+encodeURIComponent(selected)+'&index='+encodeURIComponent(selectedImage)+'&annotation='+encodeURIComponent(selectedAnnotation)+'&part='+encodeURIComponent(part);tile.append(im)}box.append(tile)}document.getElementById('previewmsg').textContent=''}else{document.getElementById('previewmsg').textContent='该类别暂无可视化图片'}updatePrediction()}""")

def images_for(cat):
    if cat not in CATS:
        return []
    save = 'saved_results_restart' if cat == 'juice_bottle' else 'saved_results'
    folder = ROOT / save / 'mvtec_loco' / cat / 'vis'
    images = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')) if folder.exists() else []
    return images

def image_scores(cat):
    path = ROOT / 'saved_results' / 'image_scores' / (cat + '.csv')
    if path.exists():
        with path.open(encoding='utf-8', newline='') as stream:
            return {row['image']: float(row['score']) for row in csv.DictReader(stream)}
    if cat == 'breakfast_box':
        path = ROOT / 'saved_results' / 'breakfast_ab' / 'scores.csv'
        if path.exists():
            with path.open(encoding='utf-8', newline='') as stream:
                return {row['defect_type'] + '/' + Path(row['image_path']).name: float(row['score_A']) for row in csv.DictReader(stream)}
    return {}

def annotations_for(cat, image):
    folder = ROOT / 'data' / 'mvtec_loco' / cat / 'ground_truth' / image.parent.name / image.stem
    return sorted(p for p in folder.glob('*.png') if p.is_file()) if folder.exists() else []

def ab_path(cat, image):
    if cat not in CATS: return None
    folder = 'breakfast_ab' if cat == 'breakfast_box' else 'ab_' + cat
    path = ROOT / 'saved_results' / folder / 'maps' / image.parent.name / (image.stem + '.npz')
    return path if path.is_file() else None

def render_panel(cat, image, annotation_name, part):
    with Image.open(image) as source:
        combined = source.convert('RGB')
        width, height = combined.size
        if width % 3: raise ValueError('Invalid saved visualization dimensions')
        panel_width = width // 3
        original = combined.crop((0, 0, panel_width, height))
        original_heat = combined.crop((panel_width, 0, 2 * panel_width, height))
    if part == 'image': return original
    if part == 'gt':
        if not annotation_name: return Image.new('RGB', original.size, 'black')
        annotation = next((p for p in annotations_for(cat, image) if p.name == annotation_name), None)
        if annotation is None: raise FileNotFoundError('Unknown annotation')
        with Image.open(annotation) as mask:
            return mask.convert('L').resize(original.size, Image.Resampling.NEAREST).point(lambda x: 255 if x > 0 else 0).convert('RGB')
    ab = ab_path(cat, image)
    if ab is None:
        if part == 'A': return original_heat
        raise FileNotFoundError('No A/B maps')
    with np.load(ab) as maps:
        if part in ('A', 'B'):
            a, b = maps['A'], maps['B']
            low = float(min(a.min(), b.min()))
            high = float(np.percentile(np.concatenate((a.ravel(), b.ravel())), 99.5))
            scale = np.clip((maps[part] - low) / max(high-low, 1e-8), 0, 1)
            color = cv2.applyColorMap(np.uint8(scale * 255), cv2.COLORMAP_JET)
            color = cv2.cvtColor(cv2.resize(color, original.size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            overlay = np.uint8(np.asarray(original) * .45 + color * .55)
            return Image.fromarray(overlay)
        if part == 'SMD':
            raw = np.clip(maps['SMD'], 0, 1)
            color = cv2.applyColorMap(np.uint8(raw * 255), cv2.COLORMAP_INFERNO)
            return Image.fromarray(cv2.cvtColor(cv2.resize(color, original.size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
        if part in ('A_binary', 'B_binary'):
            variant = part[0]
            folder = 'breakfast_ab' if cat == 'breakfast_box' else 'ab_' + cat
            info = json.loads((ROOT / 'saved_results' / folder / 'diagnosis/diagnosis.json').read_text(encoding='utf-8'))
            threshold = info['variants'][variant]['points']['0.01']['threshold']
            sizes = {'breakfast_box':(1600,1280),'juice_bottle':(800,1600),'pushpins':(1700,1000),'screw_bag':(1600,1100),'splicing_connectors':(1700,850)}
            full = cv2.resize(maps[variant], sizes[cat], interpolation=cv2.INTER_LINEAR).astype(np.float64)
            binary = np.uint8(full > threshold) * 255
            return Image.fromarray(cv2.resize(binary, original.size, interpolation=cv2.INTER_NEAREST)).convert('RGB')
    raise ValueError('Unknown panel')

def status():
    status_file = OUT / 'status.txt'
    lines = status_file.read_text(encoding='utf-8', errors='replace').splitlines() if status_file.exists() else []
    start = next((i for i in range(len(lines)-1, -1, -1) if lines[i] == 'Starting four-class evaluation after user confirmation'), 0)
    lines = lines[start:]
    current = next((line.split('Evaluating ', 1)[1] for line in reversed(lines) if line.startswith('Evaluating ')), None)
    stopped = any(line.startswith('STOPPED:') for line in lines)
    complete = 'All five evaluations completed' in lines
    four_complete = 'All four evaluations completed' in lines
    queue_file = ROOT / 'saved_results' / 'mvtec_loco' / 'training_queue_state.json'
    try:
        queue_state = json.loads(queue_file.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        queue_state = {}
    fifth_training = queue_state.get('category') == 'splicing_connectors' and queue_state.get('status') == 'running'
    fifth_failed = queue_state.get('category') == 'splicing_connectors' and queue_state.get('status') == 'failed'
    items = []
    progress = '准备中'
    for cat in CATS:
        done = f'Finished {cat}' in lines
        path = OUT / (cat + '.txt')
        text = path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''
        matches = re.findall(r'Log-iAUC:\s*([\d.]+),\s*Str-iAUC:\s*([\d.]+),\s*mean-iAUC:\s*([\d.]+)', text)
        if cat == current and not done:
            p = re.findall(r'(?:Testing|Saving vis results|Extract Aggregate Features)[^\r\n]*', text)
            if p: progress = p[-1][-180:]
        is_unstarted = cat == 'splicing_connectors' and not done and cat != current and not path.exists()
        items.append(dict(name=cat, status='已完成' if done else ('训练中' if cat == 'splicing_connectors' and fifth_training else '训练失败' if cat == 'splicing_connectors' and fifth_failed else '未训练、未测试' if is_unstarted else '测试失败' if stopped and cat == current else '测试中' if cat == current else '等待中'), metrics=matches[-1] if matches else None, preview=bool(images_for(cat))))
    if current and f'Finished {current}' in lines:
        current = None
    summary = '五类测试全部完成' if complete else '第五类训练中，完成后自动测试' if fifth_training else '第五类训练失败，请查看训练日志' if fifth_failed else '本次四类测试完成，第五类等待训练' if four_complete and not current else '测试已停止' if stopped else '准备中'
    return dict(items=items, finished=sum(x['status']=='已完成' for x in items), current=None if complete or stopped else current, state=summary, progress=progress, log='\n'.join(lines[-8:]))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == '/':
            body, content_type = PAGE.encode(), 'text/html; charset=utf-8'
        elif url.path == '/api/status':
            body, content_type = json.dumps(status(), ensure_ascii=False).encode(), 'application/json; charset=utf-8'
        elif url.path == '/api/images':
            cat = parse_qs(url.query).get('category', [''])[0]
            images = images_for(cat)
            scores = image_scores(cat)
            folder = 'breakfast_ab' if cat == 'breakfast_box' else 'ab_' + cat
            has_binary = (ROOT / 'saved_results' / folder / 'diagnosis/diagnosis.json').is_file()
            body = json.dumps([dict(index=i, name=p.parent.name + '/' + p.name, score=scores.get(p.parent.name + '/' + p.name), has_ab=bool(ab_path(cat,p)), has_binary=has_binary) for i,p in enumerate(images)], ensure_ascii=False).encode()
            content_type = 'application/json; charset=utf-8'
        elif url.path == '/api/annotations':
            query = parse_qs(url.query)
            cat = query.get('category', [''])[0]
            images = images_for(cat)
            try: index = int(query.get('index', ['0'])[0])
            except ValueError: self.send_error(400); return
            if index < 0 or index >= len(images): self.send_error(404); return
            body = json.dumps([p.name for p in annotations_for(cat, images[index])], ensure_ascii=False).encode()
            content_type = 'application/json; charset=utf-8'
        elif url.path == '/api/panel':
            query = parse_qs(url.query)
            cat = query.get('category', [''])[0]
            images = images_for(cat)
            try: index = int(query.get('index', ['0'])[0])
            except ValueError: self.send_error(400); return
            if index < 0 or index >= len(images): self.send_error(404); return
            try:
                panel = render_panel(cat, images[index], query.get('annotation', [''])[0], query.get('part', [''])[0])
            except (ValueError, FileNotFoundError): self.send_error(404); return
            output = BytesIO(); panel.save(output, format='PNG')
            body, content_type = output.getvalue(), 'image/png'
        elif url.path == '/api/preview':
            query = parse_qs(url.query)
            cat = query.get('category', [''])[0]
            images = images_for(cat)
            try: index = int(query.get('index', ['0'])[0])
            except ValueError: self.send_error(400); return
            if index < 0 or index >= len(images): self.send_error(404); return
            image = images[index]
            annotation_name = query.get('annotation', [''])[0]
            if annotation_name:
                annotation = next((p for p in annotations_for(cat, image) if p.name == annotation_name), None)
                if not annotation: self.send_error(404); return
                with Image.open(image) as source, Image.open(annotation) as mask:
                    combined = source.convert('RGB')
                    width, height = combined.size
                    panel_width = width // 3
                    if panel_width * 3 != width: self.send_error(500); return
                    mask = mask.convert('L').resize((panel_width, height), Image.Resampling.NEAREST)
                    mask = mask.point(lambda value: 255 if value > 127 else 0).convert('RGB')
                    combined.paste(mask, (2 * panel_width, 0))
                    output = BytesIO()
                    combined.save(output, format='PNG')
                    body, content_type = output.getvalue(), 'image/png'
            else:
                body, content_type = image.read_bytes(), 'image/png' if image.suffix.lower() == '.png' else 'image/jpeg'
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type', content_type); self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *args):
        pass

if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8766), Handler).serve_forever()
