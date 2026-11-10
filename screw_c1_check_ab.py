"""Gate C1 on reproducing previous full-test A/B localization metrics."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'saved_results/screw_c1'
previous=json.loads((ROOT/'saved_results/ab_screw_bag/spro_results.json').read_text())
current=json.loads((OUT/'spro_results.json').read_text())
reconstruction=json.loads((OUT/'reconstruction_verify.json').read_text())
diff={}
for variant in ('A','B'):
    diff[variant]={}
    for key in ('logical_au_spro','structural_au_spro'):
        diff[variant][key]=current['variants'][variant][key]-previous['variants'][variant][key]
passed=all(abs(x)<0.0001 for v in diff.values() for x in v.values()) and reconstruction['count']==341
result={'spro_passed':passed,'fraction_tolerance':0.0001,'differences_fraction':diff,'maps_max_abs_difference':reconstruction['max_abs_difference_from_original']}
(OUT/'ab_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if not passed:raise SystemExit('A/B mismatch: do not run C1')
