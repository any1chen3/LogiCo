"""Local read-only progress page for E3 cache files."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import html

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'saved_results/context_e3/cache_v1'

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        rows=[]
        for category,label in [('splicing_connectors','拼接连接器'),('juice_bottle','果汁瓶')]:
            for split,split_label in [('train','正常训练图'),('validation','正常验证图')]:
                total=len(list((ROOT/'data/mvtec_loco'/category/split/'good').glob('*.png')))
                done=len([p for p in (CACHE/category/split).glob('*.npz') if not p.name.endswith('.tmp.npz')])
                pct=100*done/total if total else 0
                rows.append(f'<section><strong>{html.escape(label)} · {split_label}</strong><span>{done} / {total}（{pct:.1f}%）</span><progress max="{total}" value="{done}"></progress></section>')
        result=ROOT/'saved_results/context_e3/evaluation_v1/summary.json'
        state='已生成比较汇总，可在实验记录中查看。' if result.exists() else '尚未生成完整三组比较结果。'
        content=('<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="15">'
                 '<title>E3 实验进度</title><style>body{font:18px sans-serif;margin:40px auto;max-width:720px;color:#183046}section{margin:22px 0;padding:16px;background:#f3f7fa;border-radius:10px}span{float:right}progress{width:100%;height:24px;margin-top:12px}</style>'
                 '<h1>E3 实验进度</h1><p>每 15 秒刷新。数字表示已保存的缓存图片数，不表示比较已经完成。</p>'+''.join(rows)+f'<p>{state}</p></html>').encode('utf-8')
        self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(content))); self.end_headers(); self.wfile.write(content)
    def log_message(self,*args): pass

if __name__=='__main__':
    print('E3 progress: http://127.0.0.1:8767',flush=True)
    HTTPServer(('127.0.0.1',8767),Handler).serve_forever()
