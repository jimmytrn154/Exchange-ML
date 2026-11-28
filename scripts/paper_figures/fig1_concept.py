"""Fig. 1: coverage/grouping decomposition illustrated on a real fracture CT.

Input: a bone-cropped CT/label volume written by extract_concept_case.py
(PENGWIN case 039, sacrum; reference labels only). Controlled copies follow the
intervention families of the mechanism tests; terms are computed on the whole
3D bone with the definitions of Methods II-A.
Usage: python fig1_concept.py <case.npz> <out.pdf>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import style as S
import concept_interventions as ci

S.apply()
d = np.load(sys.argv[1])
img, lab, sp = d['img'].astype(float), d['lab'].astype(int), d['spacing']
SLICE_Y = 43                       # coronal slice through all three fragments
copies = ci.interventions(lab, sp)
terms = {k: ci.terms(lab, m) for k, (m, _) in copies.items()}

# ---- 2D slice and crop -------------------------------------------------------
def sl(v):
    return v[:, SLICE_Y, :]

sx, sz = sp[0], sp[2]
ref2 = sl(lab)
zz, xx = np.where(ref2 > 0)
x0, x1 = max(xx.min() - 6, 0), min(xx.max() + 7, ref2.shape[1])
W_mm = (x1 - x0) * sx
ASPECT = 0.66                       # displayed height / width
nz = int(round(ASPECT * W_mm / sz))
zc = int(round((zz.min() + zz.max()) / 2)) + 8   # show more cranial context
z0 = int(np.clip(zc - nz // 2, 0, ref2.shape[0] - nz)); z1 = z0 + nz
crop = (slice(z0, z1), slice(x0, x1))
H_mm = nz * sz
extent = [0, W_mm, 0, H_mm]
ct = np.clip((sl(img)[crop] + 250) / 1500, 0, 1)

COLOR = {1: S.FRAG[0], 2: S.FRAG[1], 3: S.FRAG[2], 4: S.FRAG[3]}


def overlay(lab2, alpha=0.55):
    rgb = np.repeat(ct[..., None], 3, 2)
    for i, c in COLOR.items():
        m = lab2 == i
        rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(to_rgb(c))
    return rgb


def contours(ax, lab2, ids=None, **kw):
    ids = ids or [i for i in COLOR if (lab2 == i).any()]
    for i in ids:
        ax.contour((lab2 == i).astype(float), levels=[0.5], extent=extent,
                   origin='lower', **kw)


def show(ax, lab2):
    ax.imshow(overlay(lab2), extent=extent, origin='lower', interpolation='nearest')
    contours(ax, lab2, colors='white', linewidths=0.45)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# ---- layout (inches) -----------------------------------------------------------
W, Hfig = S.TEXTWIDTH, 2.05
fig = plt.figure(figsize=(W, Hfig))
iw = 0.93
ih = iw * ASPECT
top = Hfig - 0.40
gap = 0.06
xs = [0.03] + [0.03 + iw + 0.17 + k * (iw + gap) for k in range(4)]
DASH = [(0, (1.6, 1.2))]


def axes_in(x, y, w, h):
    return fig.add_axes([x / W, y / Hfig, w / W, h / Hfig])

# (a) reference map
ax = axes_in(xs[0], top - ih, iw, ih)
show(ax, sl(lab)[crop])
ax.plot([4, 34], [4, 4], color='white', lw=1.0, solid_capstyle='butt')
ax.text(19, 6.5, '30 mm', color='white', ha='center', va='bottom', fontsize=5.8)
ax.set_title('Reference', fontsize=7.5, pad=2.5, color=S.INK)
S.panel_label(ax, 'a', '', x=-0.02, y=1.0, )
ax.texts[-1].set_position((-0.03, 1.035))

# (b) controlled copies
names = [('merge', 'Merge'), ('split', 'Split'), ('removal', 'Removal'),
         ('boundary', 'Boundary shift')]
bh = 0.34
for k, (key, title) in enumerate(names):
    m2 = sl(copies[key][0])[crop]
    r2 = sl(lab)[crop]
    ax = axes_in(xs[k + 1], top - ih, iw, ih)
    show(ax, m2)
    if key in ('merge', 'removal'):       # former outline of the changed fragment
        fid = copies[key][1]['merged' if key == 'merge' else 'removed']
        contours(ax, r2, ids=[fid], colors='white', linewidths=0.7, linestyles=DASH)
    if key == 'boundary':                 # removed rim
        rim = (r2 > 0) & (m2 == 0)
        rgba = np.zeros(rim.shape + (4,)); rgba[rim] = (1, 1, 1, 1)
        ax.imshow(rgba, extent=extent, origin='lower', interpolation='nearest')
    ax.set_title(title, fontsize=7.5, pad=2.5, color=S.INK)
    if k == 0:
        ax.text(-0.10, 1.035, '$\\bf{b}$', transform=ax.transAxes, fontsize=8,
                ha='left', va='bottom')
    # readout of the three terms against the reference (whole 3D bone)
    t = terms[key]
    bx = axes_in(xs[k + 1] + 0.30, top - ih - 0.09 - bh, iw - 0.34, bh)  # same scale in all panels
    vals = [t['A'], t['wQvox'], t['wQjoint']]
    cols = [S.C_COV, S.C_GRP, S.C_GRPJ]
    ypos = [2, 1, 0]
    bx.barh(ypos, vals, height=0.66, color=cols, edgecolor='none')
    for y, v in zip(ypos, vals):
        bx.text(v + 0.015, y, f'{v:.2f}' if v > 0 else '0', va='center', ha='left',
                fontsize=6.2, color=S.INK)
    bx.set_xlim(0, 0.50); bx.set_ylim(-0.55, 2.55)
    bx.set_yticks(ypos)
    bx.set_yticklabels(['$A$', '$wQ_{\\rm vox}$', '$wQ_{\\rm joint}$'])
    bx.tick_params(axis='y', length=0, pad=1.2, labelsize=6.4)
    bx.set_xticks([])
    bx.spines['bottom'].set_visible(False)
    bx.spines['left'].set_linewidth(0.6)

fig.text((xs[0] + iw / 2) / W, (top - ih - 0.09 - bh / 2) / Hfig,
         'terms vs. reference,\ncomputed on the\nwhole 3D bone', ha='center', va='center',
         fontsize=6.2, color=S.MUTED, linespacing=1.15)

# group captions over (b)
for (a, b), txt, col in (((1, 2), 'grouping changes, foreground fixed', S.C_GRP),
                         ((3, 4), 'foreground changes, grouping fixed', S.C_COV)):
    y = top + 0.205
    fig.text((xs[a] + xs[b] + iw) / 2 / W, (y + 0.025) / Hfig, txt, ha='center',
             va='bottom', fontsize=6.6, color=col, style='italic')
    fig.add_artist(plt.Line2D([(xs[a] + 0.04) / W, (xs[b] + iw - 0.04) / W],
                              [y / Hfig, y / Hfig], color=col, lw=0.6))

# (c) scoring schematic -----------------------------------------------------------
cx0 = xs[4] + iw + 0.17
cw = W - cx0 - 0.02
Ht = Hfig - 0.06
ax = axes_in(cx0, 0.02, cw, Ht)
ax.set_xlim(0, cw); ax.set_ylim(0, Ht); ax.axis('off')
ax.text(0.0, Ht - 0.13, '$\\bf{c}$', fontsize=8, ha='left', va='center')
ax.text(0.15, Ht - 0.13, 'Prediction-only score per bone', fontsize=7.5, ha='left',
        va='center', color=S.INK)


def light(c, f=0.88):
    r = np.array(to_rgb(c)); return tuple(r + (1 - r) * f)


def box(x, y, w, h, ec, lines, size=6.4, ys=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.035',
                                lw=0.7, ec=ec, fc=light(ec)))
    n = len(lines)
    for i, s in enumerate(lines):
        yy = ys[i] if ys else y + h * (n - i - 0.5) / n
        ax.text(x + w / 2, yy, s, ha='center', va='center', fontsize=size, color=S.INK)


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=5.5,
                                 lw=0.6, color=S.MUTED, shrinkA=0, shrinkB=0))

pad = 0.02
bw = (cw - 3 * pad) / 2
y_in = Ht - 0.50
box(pad, y_in, cw - 2 * pad, 0.22, '#7F7F7F',
    ['members $i,j$: fragment maps $F_i, F_j$ of one bone'], size=6.3)
# coverage (left) and grouping (right)
y_g0, y_g1 = 0.50, y_in - 0.12
box(pad, y_g1 - 0.52, bw, 0.52, S.C_COV,
    ['Coverage', '$A=|F_i\\triangle F_j|/|U|$', '$U=F_i\\cup F_j$'], size=6.3)
xg = 2 * pad + bw
gh = y_g1 - y_g0
box(xg, y_g0, bw, gh, S.C_GRP, ['Grouping on', '$C=F_i\\cap F_j$'], size=6.3,
    ys=[y_g1 - 0.075, y_g1 - 0.185])
ax.text(xg + bw / 2, y_g0 + 0.17, '$\\mathrm{VI}=H(P|Q)+H(Q|P)$', ha='center',
        va='center', fontsize=6.0)
ax.text(xg + bw / 2, y_g0 + 0.06, '$Q_{\\rm joint}=\\mathrm{VI}/H(P,Q)$', ha='center',
        va='center', fontsize=6.0)
# contingency table of the merge copy on C (rows: reference, cols: copy)
m = copies['merge'][0]; Cm = (lab > 0) & (m > 0)
tab = np.array([[((lab == r) & (m == c) & Cm).sum() for c in (1, 2)] for r in (1, 2, 3)],
               float)
tab /= tab.sum()
tw, th = 0.34, 0.26
tx = cx0 + xg + bw / 2 - tw / 2 + 0.06
ty = 0.02 + y_g0 + 0.27
tax = fig.add_axes([tx / W, ty / Hfig, tw / W, th / Hfig])
tax.imshow(tab, cmap='Oranges', vmin=0, vmax=0.95, aspect='auto')
for (r, c), v in np.ndenumerate(tab):
    tax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=5.3,
             color='white' if v > 0.45 else S.INK)
tax.set_xticks([]); tax.set_yticks([])
for s in tax.spines.values():
    s.set_linewidth(0.4)
tax.text(-0.12, 0.5, '$P$', transform=tax.transAxes, ha='right', va='center', fontsize=6)
tax.text(0.5, 1.04, '$Q$', transform=tax.transAxes, ha='center', va='bottom', fontsize=6)
# output
y_out = 0.02
box(pad, y_out, cw - 2 * pad, 0.40, S.C_DJNT,
    ['grouping $wQ$ or combined $A+wQ$', 'mean of 3 pairs $\\rightarrow$ review order'],
    size=6.3)
arrow((pad + bw / 2, y_in), (pad + bw / 2, y_g1))
arrow((xg + bw / 2, y_in), (xg + bw / 2, y_g1))
arrow((pad + bw / 2, y_g1 - 0.52), (pad + bw / 2, y_out + 0.40))
arrow((xg + bw / 2, y_g0), (xg + bw / 2, y_out + 0.40))
ax.text(pad + bw / 2, (y_g1 - 0.52 + y_out + 0.40) / 2, '$w=|C|/|U|$', ha='center',
        va='center', fontsize=6.0, color=S.INK,
        bbox=dict(fc='white', ec='none', pad=0.6))

fig.savefig(sys.argv[2])
fig.savefig(sys.argv[2].replace('.pdf', '.png'), dpi=300)
print({k: {a: round(float(v), 3) for a, v in t.items()} for k, t in terms.items()})
