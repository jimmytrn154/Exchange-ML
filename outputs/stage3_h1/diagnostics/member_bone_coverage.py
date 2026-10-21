"""Per-bone GT coverage of a retrained member's validation predictions.

Compares the member's new predictions with its 2026-10-05 (pre-fix) predictions
and the retrained Stage 2 predictions. Coverage = fraction of each GT bone's
voxels predicted with the correct anatomy label block (1-50 sacrum, 51-100 left
hip, 101-150 right hip, 151-200 femur). Retrospective debugging only; reads GT
and is not part of acquisition scoring.

Usage: python outputs/stage3_h1/diagnostics/member_bone_coverage.py member_0X
"""
import glob
import statistics as st
import sys

import SimpleITK as sitk

mem = sys.argv[1]
P = {
    'old': f'/mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/stage3_h1/members/{mem}/validation_predictions/%s.mha',
    'stage2': 'outputs/stage2_backend/validation_predictions/%s.mha',
    'new': f'outputs/stage3_h1/members/{mem}/validation_predictions/%s.mha',
}
rd = lambda p: sitk.GetArrayFromImage(sitk.ReadImage(p))
res = {k: {} for k in P}
low = []
for f in sorted(glob.glob(P['new'] % '*')):
    c = f.split('/')[-1][:3]
    gt = rd(glob.glob(f'dataset/PENGWIN26_task1_2/*/{c}/label.mha')[0])
    for tag, path in P.items():
        v = rd(path % c)
        for g, name in enumerate(('Sac', 'LHip', 'RHip', 'Femur')):
            m = (gt > 0) & ((gt - 1) // 50 == g)
            if m.sum() == 0:
                continue
            lab = (((v - 1) // 50 == g) & (v > 0) & m).sum() / m.sum()
            res[tag].setdefault(name, []).append(lab)
            if tag == 'new' and lab < 0.9:
                low.append((c, name, round(float(lab), 3)))
for tag in P:
    print(tag, {b: (len(x), round(st.median(x), 3), sum(a < 0.5 for a in x), sum(a < 0.9 for a in x))
                for b, x in res[tag].items()})
print('format: bone -> (n, median correct-label coverage, cases < 0.5, cases < 0.9)')
print('new < 0.9:', low)
