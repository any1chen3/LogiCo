"""Export E4 saved per-window intermediate masks as inspectable PNG files."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]/'saved_results/context_e4_window_intersection'


def main():
    for sample in ('connector_missing_026','connector_good_000','breakfast_064'):
        folder=ROOT/sample
        info=json.loads((folder/'matches.json').read_text(encoding='utf-8'))
        windows=info['windows']['intersects']
        dest=folder/'intermediate_png';dest.mkdir(exist_ok=True)
        with np.load(folder/'window_masks.npz') as z:
            Image.fromarray(np.uint8(z['raw_observed'])*255).save(dest/'01_raw_observed.png')
            for i,window in enumerate(windows):
                name=window['window']
                full=np.uint8(z['intersects_observed_inputs'][i])*255
                # One clipped binary vote per reference; gray means fractional support.
                remaining=np.uint8(np.rint(z['intersects_missing_after_clip'][i].mean(axis=0)*255))
                Image.fromarray(full).save(dest/f'{name}_02_matching_input_full.png')
                Image.fromarray(remaining).save(dest/f'{name}_03_remaining_vote.png')
        print(sample,'exported',len(windows),'window pairs',flush=True)


if __name__=='__main__':main()
