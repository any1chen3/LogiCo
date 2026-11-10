"""Run the remaining LOCO categories after the current training finishes."""
import ctypes
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'saved_results' / 'mvtec_loco'
STATUS = RESULTS / 'training_queue_status.txt'
CURRENT_PID = 49024
CATEGORIES = ('pushpins', 'screw_bag', 'splicing_connectors')
STATE = RESULTS / 'training_queue_state.json'

def state(category, batch, status):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(category=category, batch=batch, status=status)), encoding='utf-8')
    temporary.replace(STATE)

def note(message):
    line = f'{datetime.now():%Y-%m-%d %H:%M:%S} {message}\n'
    with STATUS.open('a', encoding='utf-8') as out:
        out.write(line)

def wait_for_process(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), 'Cannot monitor current training process')
    try:
        kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
    finally:
        kernel.CloseHandle(handle)

def run_category(category, batch):
    folder = RESULTS / category
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f'console_batch_{batch}.txt'
    command = [sys.executable, '-u', str(ROOT / 'train_logico.py'), '--dataset', 'mvtec_loco', '--category', category, '--save_dir', 'saved_results', '--batch_size', str(batch)]
    note(f'{category}: starting batch_size={batch}; output={output}')
    state(category, batch, 'running')
    with output.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    text = output.read_text(encoding='utf-8', errors='replace')
    return result.returncode, 'out of memory' in text.lower()

if __name__ == '__main__':
    note('Starting the three remaining categories sequentially')
    for category in CATEGORIES:
        code, oom = run_category(category, 2)
        if code and oom:
            note(f'{category}: GPU out of memory at batch_size=2; restarting at batch_size=1')
            code, _ = run_category(category, 1)
        if code or not (RESULTS / category / 'LogiCo_epoch_200_ckpt.pkl').is_file():
            note(f'STOPPED: {category} did not complete (exit={code}); inspect console output')
            state(category, 1 if oom else 2, 'failed')
            sys.exit(1)
        note(f'{category}: completed')
    note('All five categories completed')
    state(category, 1 if oom else 2, 'completed')
