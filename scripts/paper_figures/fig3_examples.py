"""Fig. 3: two held-out bones where the two components disagree.

Rows: the E5 qualitative selections 'partition_detects_structure' (176/right hip)
and 'presence_detects_missing' (160/sacrum), selected by the E5 rule, not by hand.
Overlays are recovered from the saved E5 renderings (qualitative_masks.py); scores
and ranks come from heldout/scores.csv.
Usage: python fig3_examples.py <outputs/stage3_h1_v2> <out.pdf>
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import style as S
import qualitative_masks as Q

S.apply()
root, out = sys.argv[1], sys.argv[2]
fig_dir = f'{root}/e5_supplement_v1/figures'
scores = list(csv.DictReader(open(f'{root}/heldout/scores.csv')))


def rank(case, bone, key):
    v = {(r['case_id'], r['bone']): float(r[key]) for r in scores}
    x = v[(case, bone)]
    return 1 + sum(o > x for o in v.values()), len(v), x

ROWS = [
    dict(png='qualitative_partition_detects_structure.png', y=636, case='176',
         bone='right_hip', crop=(slice(0, 470), slice(170, 594)),
         label='Right hip'),
    dict(png='qualitative_presence_detects_missing.png', y=638, case='160',
         bone='sacrum', crop=(slice(70, 450), slice(0, 594)),
         label='Sacrum'),
]
COLS = ['Reference', 'Member 1', 'Member 2', 'Member 3']
CLASS_COLOR = {1: S.FRAG[0], 2: S.FRAG[1], 3: S.FRAG[2]}

data = []
for r in ROWS:
    ct, maps, names, _ = Q.recover(f'{fig_dir}/{r["png"]}', r['y'])
    ct = ct[r['crop']]
    maps = [m[r['crop']] for m in maps]
    data.append((ct, maps))

W = S.COLWIDTH
left, gap, right = 0.17, 0.035, 0.01
pw = (W - left - right - 3 * gap) / 4
heights = [pw * d[0].shape[0] / d[0].shape[1] for d in data]
txt_h, title_h = 0.30, 0.16
Hfig = title_h + sum(heights) + 2 * txt_h - 0.08
fig = plt.figure(figsize=(W, Hfig))


def render(ax, ct, lab):
    rgb = np.repeat(ct[..., None], 3, 2)
    for k, c in CLASS_COLOR.items():
        m = lab == k
        rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(to_rgb(c))
    ax.imshow(rgb, interpolation='lanczos')
    for k in CLASS_COLOR:
        if (lab == k).any():
            ax.contour((lab == k).astype(float), levels=[0.5], colors='white',
                       linewidths=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

y = Hfig - title_h
for i, (r, (ct, maps), h) in enumerate(zip(ROWS, data, heights)):
    y -= h
    for j, lab in enumerate(maps[:1] + maps[1:]):
        ax = fig.add_axes([(left + j * (pw + gap)) / W, y / Hfig, pw / W, h / Hfig])
        render(ax, ct, lab)
        if i == 0:
            ax.set_title(COLS[j], fontsize=7.3, pad=2.5)
    fig.text(0.05 / W, (y + h / 2) / Hfig, f'$\\bf{{{"ab"[i]}}}$  {r["label"]}',
             rotation=90, ha='center', va='center', fontsize=7.5)
    ra = rank(r['case'], r['bone'], 'presence')
    rq = rank(r['case'], r['bone'], 'partition_v2')
    re = rank(r['case'], r['bone'], 'mean_entropy')
    fmt = lambda t: 'tied lowest' if t[2] == 0 else f'rank {t[0]}'
    line = (f'$A={ra[2]:.3f}$ ({fmt(ra)}/{ra[1]})    '
            f'$wQ_{{\\rm joint}}={rq[2]:.3f}$ ({fmt(rq)})    '
            f'entropy {fmt(re)}')
    fig.text((left + (W - left - right) / 2) / W, (y - 0.035) / Hfig, line,
             ha='center', va='top', fontsize=6.6, color=S.INK)
    print(r['case'], r['bone'], 'A', ra, 'wQj', rq, 'entropy', re)
    y -= txt_h

fig.savefig(out)
fig.savefig(out.replace('.pdf', '.png'), dpi=300)
