"""Wait for screw_bag training, then evaluate the four completed classes."""
import json
import subprocess
import sys
from pathlib import Path
from train_remaining_queue import wait_for_process

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'saved_results' / 'four_class_evaluation'
STATE = ROOT / 'saved_results' / 'mvtec_loco' / 'training_queue_state.json'
CATEGORIES = [('breakfast_box', 'saved_results'), ('juice_bottle', 'saved_results_restart'), ('pushpins', 'saved_results'), ('screw_bag', 'saved_results')]

def write_status(message):
    with (OUT / 'status.txt').open('a', encoding='utf-8') as file:
        file.write(message + '\n')

def dashboard(category, status):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(category=category, batch=2, status=status)), encoding='utf-8')
    temporary.replace(STATE)

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    write_status('Starting four-class evaluation after user confirmation')
    ckpt = ROOT / 'saved_results' / 'mvtec_loco' / 'screw_bag' / 'LogiCo_epoch_200_ckpt.pkl'
    if not ckpt.is_file():
        write_status('STOPPED: screw_bag did not complete; no evaluation started')
        dashboard('screw_bag', 'training failed')
        sys.exit(1)
    for category, save_dir in CATEGORIES:
        checkpoint = ROOT / save_dir / 'mvtec_loco' / category / 'LogiCo_best_ckpt.pkl'
        if not checkpoint.is_file():
            write_status(f'STOPPED: missing best checkpoint for {category}')
            dashboard(category, 'missing checkpoint')
            sys.exit(1)
        write_status(f'Evaluating {category}')
        dashboard(category, 'evaluating')
        args = [sys.executable, '-u', str(ROOT / 'evaluation.py'), '--dataset', 'mvtec_loco', '--category', category, '--save_dir', save_dir, '--batch_size', '2']
        with (OUT / (category + '.txt')).open('w', encoding='utf-8') as file:
            result = subprocess.run(args, cwd=ROOT, stdout=file, stderr=subprocess.STDOUT)
        if result.returncode:
            write_status(f'STOPPED: {category} failed with exit code {result.returncode}; inspect {category}.txt')
            dashboard(category, 'evaluation failed')
            sys.exit(result.returncode)
        write_status(f'Finished {category}')
    write_status('All four evaluations completed')
    dashboard('screw_bag', 'four evaluations completed')
