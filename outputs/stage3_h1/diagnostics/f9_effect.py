"""F9 effect: new (F9) vs archived pre-F9 validation predictions, per run.

For each run: cases whose label map changed, whether a RightHip bone-skeleton
fallback fired there (from the F9 predict log), GT femur coverage before/after,
and whether the foreground-probability maps changed (F9 should not touch them).
Retrospective debugging only; reads GT.

Usage: python outputs/stage3_h1/diagnostics/f9_effect.py
"""
import glob
import re

import numpy as np
import SimpleITK as sitk

SA = '/mnt/sdb/shared/chuong/archive_2026-10-20_pre_F9'
RUNS = {
    'stage2': ('outputs/stage2_backend', f'{SA}/stage2_validation_predictions', None,
               'outputs/logs/c0_stage2_predict_f9.log'),
}
for m in ('01', '02', '03'):
    RUNS[f'member_{m}'] = (f'outputs/stage3_h1/members/member_{m}',
                           f'{SA}/m{m}_infer/validation_predictions',
                           f'{SA}/m{m}_infer/validation_probabilities',
                           f'outputs/logs/c{m[1]}_member_{m}_predict_f9.log')
rd = lambda p: sitk.GetArrayFromImage(sitk.ReadImage(p))


def fallback_cases(log):
    cases, pending = set(), False
    for line in open(log, errors='replace'):
        if 'bone-skeleton fallback (' in line:
            pending = True
        m = re.search(r'write_label_map.*?(\d{3})\.mha', line)
        if m:
            if pending:
                cases.add(m.group(1))
            pending = False
    return cases


for run, (root, old_pred, old_prob, log) in RUNS.items():
    fb = fallback_cases(log)
    changed, prob_changed, rows = [], 0, []
    for f in sorted(glob.glob(f'{root}/validation_predictions/*.mha')):
        c = f.split('/')[-1][:3]
        new, old = rd(f), rd(f'{old_pred}/{c}.mha')
        if not np.array_equal(new, old):
            changed.append(c)
            gt = rd(glob.glob(f'dataset/PENGWIN26_task1_2/*/{c}/label.mha')[0])
            m = gt > 150
            cov = lambda v: ((v > 150) & m).sum() / max(1, m.sum())
            rows.append(f'{c}{"*" if c in fb else ""} femur {cov(old):.3f}->{cov(new):.3f} '
                        f'voxels changed {int((new != old).sum())}')
        if old_prob is not None:
            if not np.array_equal(rd(f'{root}/validation_probabilities/{c}.mha'), rd(f'{old_prob}/{c}.mha')):
                prob_changed += 1
    print(f'{run}: fallback cases {sorted(fb)}; changed {len(changed)}; '
          f'changed without fallback {sorted(set(changed) - fb)}; probability maps changed {prob_changed}')
    for r in rows:
        print('   ', r)
print('* = RightHip bone-skeleton fallback fired in the F9 run')
