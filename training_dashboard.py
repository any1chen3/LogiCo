"""Local, read-only progress page for the juice_bottle training run."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data' / 'mvtec_loco_cached-dinov3_vitb16-[11]-8-512' / 'juice_bottle'
DATA = ROOT / 'data' / 'mvtec_loco' / 'juice_bottle'
LOG = ROOT / 'saved_results_restart' / 'mvtec_loco' / 'juice_bottle' / 'log.txt'
PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LogiCo 训练进度</title>
<style>body{font:16px system-ui;background:#f2f5fa;color:#1c2940;max-width:800px;margin:50px auto;padding:0 20px}section{background:white;border-radius:14px;padding:22px;margin:16px 0;box-shadow:0 4px 18px #1c294010}progress{width:100%;height:22px}pre{white-space:pre-wrap;word-break:break-all;font-size:13px}small{color:#62748b}</style>
<h1>LogiCo 训练进度</h1><small id="run">读取中…</small>
<section><h2>特征提取</h2><div id="features"></div></section>
<section><h2>模型训练</h2><div id="epoch">读取中…</div><progress id="epbar" max="200" value="0"></progress><p id="gpu"></p><small>训练进度仅在每个 epoch 结束后更新；显卡占用包括其他程序。</small></section>
<section><h2>最新日志</h2><pre id="log"></pre><small id="updated"></small></section>
<script>async function update(){try{let r=await fetch('/status',{cache:'no-store'}),s=await r.json();document.getElementById('run').textContent=`${s.category} · 批次 ${s.batch} · ${s.status} · 页面每 5 秒刷新`;document.getElementById('features').innerHTML=s.phases.map(p=>`<p>${p.name}：${p.done} / ${p.total}</p><progress max="${p.total||1}" value="${p.done}"></progress>`).join(''); document.getElementById('epoch').textContent=s.epoch?`已完成 ${s.epoch} / 200 个 epoch`:'尚未完成第 1 个 epoch';document.getElementById('epbar').value=s.epoch;document.getElementById('gpu').textContent=s.gpu;document.getElementById('log').textContent=s.log;document.getElementById('updated').textContent='页面更新时间：'+new Date().toLocaleTimeString()+'；请结合日志判断训练是否仍在运行。'}catch(e){document.getElementById('updated').textContent='无法连接进度服务：'+e}} update();setInterval(update,5000)</script></html>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            state_file = ROOT / 'saved_results' / 'mvtec_loco' / 'training_queue_state.json'
            state = json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else dict(category='pushpins', batch=2, status='starting')
            category = state['category']
            data = ROOT / 'data' / 'mvtec_loco' / category
            cache = ROOT / 'data' / 'mvtec_loco_cached-dinov3_vitb16-[11]-8-512' / category
            result_dir = 'saved_results_restart' if category == 'juice_bottle' else 'saved_results'
            log_path = ROOT / result_dir / 'mvtec_loco' / category / 'log.txt'
            phases = []
            for name in ('train', 'test', 'validation'):
                total = sum(1 for p in (data / name).rglob('*') if p.is_file() and p.suffix in ('.png', '.JPG', '.bmp'))
                done = sum(1 for p in (cache / name).rglob('*.pkl') if p.is_file())
                phases.append(dict(name=name, total=total, done=min(done, total)))
            log = log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else ''
            starts = [m.start() for m in re.finditer(r"\[train_logico.py\]\[line:\s*34\]", log)]
            if starts:
                log = log[starts[-1]:]
            epochs = re.findall(r'epoch \[(\d+)/200\]', log)
            try:
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
            except Exception:
                gpu = '显卡状态暂不可用'
            body = json.dumps(dict(phases=phases, epoch=int(epochs[-1]) if epochs else 0, gpu=gpu, log='\n'.join(log.splitlines()[-8:]), **state), ensure_ascii=False).encode()
            kind = 'application/json; charset=utf-8'
        elif self.path == '/':
            body, kind = PAGE.encode(), 'text/html; charset=utf-8'
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type', kind); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *args):
        pass

if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
