"""Per-bone GT coverage of the retrained Stage 2 final predictions vs the pre-fix archive."""
import glob, statistics as st
import numpy as np, SimpleITK as sitk
NEW = 'outputs/stage2_backend/validation_predictions/%s.mha'
OLD = 'outputs/archive/2026-10-12_pre_backend_fix/stage2_backend/validation_predictions/%s.mha'
rd = lambda p: sitk.GetArrayFromImage(sitk.ReadImage(p))
res = {'old': {}, 'new': {}}
for f in sorted(glob.glob(NEW % '*')):
    c = f.split('/')[-1][:3]
    gt = rd(glob.glob(f'dataset/PENGWIN26_task1_2/*/{c}/label.mha')[0])
    if gt.max() > 150:
        continue  # femur-only scan
    for tag, path in (('old', OLD), ('new', NEW)):
        v = rd(path % c); fg = v > 0
        for g, name in enumerate(('Sac', 'LHip', 'RHip')):
            m = (gt > 0) & ((gt - 1) // 50 == g)
            any_fg = (fg & m).sum() / m.sum()
            lab = (fg & ((v - 1) // 50 == g) & m).sum() / m.sum()
            res[tag].setdefault(name, []).append((any_fg, lab))
for tag in ('old', 'new'):
    print(tag, {b: (len(x), round(st.median(a for a, _ in x), 3), round(st.median(l for _, l in x), 3),
                    sum(a < 0.5 for a, _ in x)) for b, x in res[tag].items()})
print('format: bone -> (n pelvic cases, median any-fg coverage, median correct-label coverage, cases any-fg < 0.5)')
