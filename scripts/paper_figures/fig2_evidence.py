"""Fig. 2: mechanism, normalization and held-out ranking evidence.

Reads saved E1-E3 artifacts only (no experiments are rerun).
Usage: python fig2_evidence.py <outputs/stage3_h1_v2> <out.pdf>
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import style as S
from aurc_check import load, curve, aurc

S.apply()
root, out = sys.argv[1], sys.argv[2]

# ---- data -------------------------------------------------------------------
resp = [r for r in csv.DictReader(open(f'{root}/perturbations/responses.csv'))
        if r['eligible'] == 'True']
OPS = [('identity', 'Unchanged'), ('permutation', 'Relabeled'), ('merge', 'Merge'),
       ('split', 'Split'), ('missing', 'Removal'), ('boundary', 'Boundary')]
frac = {}
for op, _ in OPS:
    rr = [r for r in resp if r['operation'] == op]
    frac[op] = (len(rr), np.mean([float(r['presence']) > 0 for r in rr]),
                np.mean([float(r['weighted_q_joint']) > 0 for r in rr]))
assert sum(v[0] for k, v in frac.items() if k not in ('identity', 'permutation')) == 372

dev = list(csv.DictReader(open(f'{root}/validation/scores.csv')))
sh_v = np.array([float(r['partition_share_v1']) for r in dev]) * 100
sh_j = np.array([float(r['partition_share_v2']) for r in dev]) * 100

keys, sc, rk = load('heldout', root)
risk = [float(rk[k]['member_mean_merge_split']) for k in keys]
CURVES = [  # key, label, colour, linestyle, linewidth
    ('mean_entropy', 'Mean entropy', S.C_ENT, '-', 1.0),
    ('presence', 'Coverage $A$', S.C_COV, '-', 1.1),
    ('fa_v1', '$D_{\\rm vox}$', S.C_DVOX, (0, (3, 1.5)), 1.1),
    ('fa_v2', '$D_{\\rm joint}$', S.C_DJNT, '-', 1.1),
    ('partition_v1', '$wQ_{\\rm vox}$', S.C_GRP, '-', 1.4),
]
curves = {}
for key, *_ in CURVES:
    c, r = curve([float(sc[k][key]) for k in keys], risk)
    curves[key] = (c, r, aurc(c, r))
rand = np.mean(risk) * (1 - 1 / len(risk))      # expected AURC of a random order

cmp = [r for r in csv.DictReader(open(f'{root}/heldout/comparison.csv'))
       if r['stratum'] == 'pooled' and r['risk_source'] == 'member_mean']
rho = {(r['score'], r['risk']): float(r['rho']) for r in cmp}
ENDS = [('merge_split', 'M+S'), ('one_minus_f1', '$1-F_1$'),
        ('missing_volume', 'Miss. vol.')]

# ---- layout -------------------------------------------------------------------
W, H = S.TEXTWIDTH, 1.98
fig = plt.figure(figsize=(W, H))
B, T = 0.36, H - 0.30
def ax_in(x, w, b=B, t=T):
    return fig.add_axes([x / W, b / H, w / W, (t - b) / H])


def title(x, letter, text):
    fig.text(x / W, (H - 0.04) / H, f'$\\bf{{{letter}}}$  {text}', ha='left', va='top',
             fontsize=8, color=S.INK)

# (a) controlled responses: nonzero fraction per term
ax = ax_in(0.64, 0.70)
for i, (op, name) in enumerate(OPS):
    n, fa, fq = frac[op]
    for j, (f, col) in enumerate(((fa, S.C_COV), (fq, S.C_GRP))):
        on = f > 0.5
        ax.add_patch(plt.Rectangle((j + 0.06, i + 0.08), 0.88, 0.84,
                                   fc=col if on else '#F2F2F2', ec='none'))
        ax.text(j + 0.5, i + 0.5, f'{100 * f:.0f}%', ha='center', va='center',
                fontsize=6.4, color='white' if on else S.MUTED)
    ax.text(-0.08, i + 0.5, f'{name} ({n})', ha='right', va='center', fontsize=6.6)
ax.set_xlim(0, 2); ax.set_ylim(len(OPS), 0)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['$A$', '$wQ_{\\rm joint}$'])
ax.xaxis.tick_top(); ax.tick_params(axis='x', length=0, pad=1.5)
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
title(0.02, 'a', 'Controlled responses')
ax.text(0.4, len(OPS) + 0.25, 'interventions with nonzero term', ha='center', va='top',
        fontsize=6.2, color=S.MUTED)

# (b) grouping share of the combined score
ax = ax_in(1.74, 0.80)
rng = np.random.default_rng(7)
jit = rng.uniform(-0.13, 0.13, len(sh_v))
for a, b, jj in zip(sh_v, sh_j, jit):
    ax.plot([0 + jj, 1 + jj], [a, b], color='#BDBDBD', lw=0.35, alpha=0.6, zorder=1)
for xi, (v, col) in enumerate(((sh_v, S.C_DVOX), (sh_j, S.C_DJNT))):
    ax.scatter(xi + jit, v, s=5, color=col, lw=0, alpha=0.85, zorder=2)
    md = np.median(v)
    ax.plot([xi - 0.26, xi + 0.26], [md, md], color=S.INK, lw=1.2, zorder=3)
    ax.text(xi, 103, f'{md:.1f}%', ha='center', va='bottom', fontsize=6.6)
ax.set_xlim(-0.5, 1.5); ax.set_ylim(-3, 100)
ax.set_xticks([0, 1]); ax.set_xticklabels(['$D_{\\rm vox}$', '$D_{\\rm joint}$'])
ax.set_yticks([0, 50, 100]); ax.set_ylabel('Grouping share $wQ/D$ (%)', labelpad=1)
ax.tick_params(axis='x', length=0)
ax.spines['bottom'].set_visible(False)
title(1.40, 'b', 'Normalization')
ax.text(0.5, -0.20, '80 development bones; median', transform=ax.transAxes,
        ha='center', va='top', fontsize=6.2, color=S.MUTED)

# (c) held-out risk-coverage
ax = ax_in(2.95, 1.36)
ax.axhline(np.mean(risk), color=S.C_RAND, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
for key, lab_, col, ls, lw in CURVES:
    c, r, a = curves[key]
    ax.plot(c, r, color=col, ls=ls, lw=lw, label=f'{lab_}  {a:.3f}',
            zorder=3 if key == 'partition_v1' else 2, solid_capstyle='round')
h, l = ax.get_legend_handles_labels()
h.append(plt.Line2D([], [], color=S.C_RAND, lw=0.7, ls=(0, (1, 1.5))))
l.append(f'Random  {rand:.3f}')
order = np.argsort([float(x.split()[-1]) for x in l])
ax.legend([h[i] for i in order], [l[i] for i in order], loc='center left',
          bbox_to_anchor=(1.01, 0.5), handlelength=1.6, handletextpad=0.5,
          labelspacing=0.22, title='AURC $\\downarrow$', title_fontsize=6.6,
          alignment='left')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(['0', '', '0.5', '', '1'])
ax.set_yticks([0, 0.5, 1.0])
ax.set_xlabel('Fraction of bones retained', labelpad=1)
ax.set_ylabel('Retained merge/split risk', labelpad=1)
ax.grid(axis='y', color=S.GRID, lw=0.5); ax.set_axisbelow(True)
title(2.70, 'c', 'Held-out ranking, 200 bones')

# (d) Spearman correlation with each endpoint
ax = ax_in(5.76, 1.34)
nb = len(CURVES); bw = 0.15
for g, (ek, _) in enumerate(ENDS):
    for i, (key, _, col, _, _) in enumerate(CURVES):
        v = rho[(key, ek)]
        ax.bar(g + (i - (nb - 1) / 2) * bw, v, width=bw * 0.86, color=col, lw=0)
ax.axhline(0, color=S.INK, lw=0.6)
ax.set_xticks(range(len(ENDS))); ax.set_xticklabels([e[1] for e in ENDS])
ax.tick_params(axis='x', length=0)
ax.set_ylim(-0.2, 0.8); ax.set_yticks([0, 0.4, 0.8])
ax.set_ylabel('Spearman $\\rho$ with risk', labelpad=1)
ax.spines['bottom'].set_visible(False)
ax.grid(axis='y', color=S.GRID, lw=0.5); ax.set_axisbelow(True)
title(5.40, 'd', 'Endpoint alignment')
ax.text(0.5, -0.20, 'colors as in (c)', transform=ax.transAxes, ha='center',
        va='top', fontsize=6.2, color=S.MUTED)

fig.savefig(out)
fig.savefig(out.replace('.pdf', '.png'), dpi=300)
print({k: round(v[2], 4) for k, v in curves.items()}, 'random', round(rand, 4))
print({k: v for k, v in frac.items()})
print('median share', np.median(sh_v), np.median(sh_j))
