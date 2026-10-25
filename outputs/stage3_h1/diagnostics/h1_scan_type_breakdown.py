"""H1 post-hoc check: do the GT-free scores mainly separate pelvic from femur-only scans?

Reads the frozen score CSV and the frozen Stage 2 risk CSV (retrospective only;
no score or rule is changed). Prints per-scan-type means, within-type Spearman
correlations, FA-IPD vs foreground-Jaccard agreement, and the NVI share of FA-IPD.
Scan type: case_id < 200 = pelvic, >= 200 = femur-only (validation split layout).

Usage: python outputs/stage3_h1/diagnostics/h1_scan_type_breakdown.py
"""
import csv
import statistics as st

from scipy.stats import spearmanr

S = {r['case_id']: r for r in csv.DictReader(open('outputs/stage3_h1/uncertainty_scores.csv'))}
R = {r['case_id']: r for r in csv.DictReader(open('outputs/stage2_backend/validation_metrics.csv'))
     if r['connectivity'] == '26'}
cases = sorted(S)
pel = [c for c in cases if int(c) < 200]
fem = [c for c in cases if int(c) >= 200]
risk = {
    '1-F1': lambda c: 1 - float(R[c]['instance_f1']),
    'norm-M+S': lambda c: (int(R[c]['merge_errors']) + int(R[c]['split_errors']))
                          / max(int(R[c]['retained_gt_instance_count']), 1),
    '1-Dice': lambda c: 1 - float(R[c]['binary_dice']),
}
scores = ['mean_entropy', 'p95_entropy', 'foreground_jaccard_disagreement', 'union_foreground_nvi',
          'hungarian_structural_disagreement', 'fragment_count_variance', 'fa_ipd']
print('group means (pelvic / femur-only)')
for k, f in risk.items():
    print(f'  risk {k:12s} {st.mean(f(c) for c in pel):8.3f} / {st.mean(f(c) for c in fem):8.3f}')
for s in scores:
    print(f'  score {s:34s} {st.mean(float(S[c][s]) for c in pel):.4f} / {st.mean(float(S[c][s]) for c in fem):.4f}')
print('top-8 by FA-IPD:', sorted(cases, key=lambda c: -float(S[c]['fa_ipd']))[:8])
print('\nSpearman rho: all 40 | pelvic 20 | femur-only 20')
for s in scores:
    row = []
    for k, f in risk.items():
        vals = [spearmanr([float(S[c][s]) for c in g], [f(c) for c in g])[0] for g in (cases, pel, fem)]
        row.append(f'{k} ' + ' | '.join(f'{v:+.2f}' for v in vals))
    print(f'  {s:34s} ' + '   '.join(row))
fa = [float(S[c]['fa_ipd']) for c in cases]
jac = [float(S[c]['foreground_jaccard_disagreement']) for c in cases]
share = [float(S[c]['fa_weighted_partition_contribution']) / max(1e-12, float(S[c]['fa_ipd'])) for c in cases]
print(f'\nFA-IPD vs Jaccard Spearman {spearmanr(fa, jac)[0]:.3f}; NVI term share of FA-IPD median '
      f'{st.median(share):.3%} max {max(share):.3%}')


def disagreement_location():
    """Where the three members disagree on foreground: inside vs outside GT-labelled bone."""
    import glob
    import SimpleITK as sitk
    rd = lambda p: sitk.GetArrayFromImage(sitk.ReadImage(p))
    for name, group in (('pelvic', pel), ('femur-only', fem)):
        fracs = []
        for c in group:
            gt = rd(glob.glob(f'dataset/PENGWIN26_task1_2/*/{c}/label.mha')[0]) > 0
            fg = [rd(f'outputs/stage3_h1/members/member_0{m}/validation_predictions/{c}.mha') > 0 for m in (1, 2, 3)]
            dis = (fg[0] | fg[1] | fg[2]) & ~(fg[0] & fg[1] & fg[2])
            fracs.append((dis & ~gt).sum() / max(1, dis.sum()))
        print(f'{name}: share of member foreground disagreement outside GT-labelled bone, '
              f'median {st.median(fracs):.3f} (min {min(fracs):.3f}, max {max(fracs):.3f})')


if __name__ == '__main__':
    disagreement_location()
