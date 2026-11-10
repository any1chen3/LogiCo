"""Run four paired experiments sequentially and record stage and failures."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'saved_results/ab_remaining'
CATEGORIES=('juice_bottle','pushpins','screw_bag','splicing_connectors')
EXPECTED={'juice_bottle':99.21,'pushpins':98.42,'screw_bag':88.75,'splicing_connectors':84.38}
STAGES=(('inference','breakfast_ab_inference.py'),('spro','breakfast_ab_spro.py'),('diagnosis','breakfast_ab_diagnosis.py'),('verify','breakfast_ab_verify_points.py'),('montage','breakfast_ab_montages.py'))

def status(category,stage,state,details=''):
    obj={'timestamp':datetime.now().isoformat(timespec='seconds'),'category':category,'stage':stage,'state':state,'details':details}
    temporary=BASE/'status.tmp'
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(BASE/'status.json')
    with (BASE/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    print(json.dumps(obj,ensure_ascii=False),flush=True)

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['MPLCONFIGDIR']=str(BASE/'matplotlib')
    for cat in CATEGORIES:
        env['LOGICO_AB_CATEGORY']=cat
        folder=ROOT/'saved_results'/('ab_'+cat)
        folder.mkdir(parents=True,exist_ok=True)
        for stage,script in STAGES:
            if stage=='inference' and (folder/'inference.json').exists() and (folder/'scores.csv').exists():
                data=json.loads((folder/'inference.json').read_text(encoding='utf-8'))
                auc=data['image_metrics']['Log-iAUC']*100
                if abs(auc-EXPECTED[cat])>0.0051:
                    status(cat,stage,'failed',f'existing iAUROC={auc:.6f} differs from expected {EXPECTED[cat]:.2f}')
                    return 1
                status(cat,stage,'completed',f'reused completed export; logical iAUROC={auc:.6f}%')
                continue
            status(cat,stage,'running')
            path=folder/(stage+'.log')
            with path.open('w',encoding='utf-8') as out:
                process=subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
            if process.returncode:
                status(cat,stage,'failed',f'exit={process.returncode}; see {path}')
                return process.returncode
            if stage=='inference':
                data=json.loads((folder/'inference.json').read_text(encoding='utf-8'))
                auc=data['image_metrics']['Log-iAUC']*100
                if abs(auc-EXPECTED[cat])>0.0051:
                    status(cat,stage,'failed',f'iAUROC={auc:.6f} differs from original {EXPECTED[cat]:.2f}')
                    return 1
                status(cat,stage,'completed',f'logical iAUROC={auc:.6f}% matches {EXPECTED[cat]:.2f}%')
            else: status(cat,stage,'completed')
        status(cat,'all','completed')
    status('all','all','completed')
    return 0

if __name__=='__main__':sys.exit(main())
