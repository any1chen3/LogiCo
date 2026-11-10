"""Train and evaluate the final LOCO category, sequentially."""
from pathlib import Path
import subprocess
import sys
from train_remaining_queue import run_category, note, state, RESULTS, ROOT

category = 'splicing_connectors'
output_dir = ROOT / 'saved_results' / 'four_class_evaluation'
output_dir.mkdir(parents=True, exist_ok=True)

def evaluation_note(message):
    with (output_dir / 'status.txt').open('a', encoding='utf-8') as stream:
        stream.write(message + '\n')

if __name__ == '__main__':
    note('Starting final category training and evaluation')
    batch = 2
    code, oom = run_category(category, batch)
    if code and oom:
        batch = 1
        note(f'{category}: GPU out of memory; restarting with batch_size=1')
        code, _ = run_category(category, batch)
    if code or not (RESULTS / category / 'LogiCo_epoch_200_ckpt.pkl').is_file():
        note(f'STOPPED: {category} training did not complete (exit={code})')
        state(category, batch, 'failed')
        sys.exit(1)
    note(f'{category}: training completed; starting evaluation')
    state(category, batch, 'evaluating')
    evaluation_note(f'Evaluating {category}')
    command = [sys.executable, '-u', str(ROOT / 'evaluation.py'), '--dataset', 'mvtec_loco', '--category', category, '--save_dir', 'saved_results', '--batch_size', str(batch)]
    with (output_dir / (category + '.txt')).open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        evaluation_note(f'STOPPED: {category} evaluation failed with exit code {result.returncode}')
        state(category, batch, 'evaluation failed')
        sys.exit(result.returncode)
    evaluation_note(f'Finished {category}')
    evaluation_note('All five evaluations completed')
    state(category, batch, 'completed')
    note(f'{category}: training and evaluation completed')
