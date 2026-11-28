"""Recover display masks from the saved E5 qualitative renderings.

The E5 figures (outputs/stage3_h1_v2/e5_supplement_v1/figures/*.png) show, for each
selected held-out bone, the CT slice and alpha-blended overlays of the reference
and of each ensemble member. Raw predictions are stored on the compute server; this
module re-derives the per-pixel overlay class from the rendering so that the
examples can be redrawn in the manuscript style. Overlay colors are fitted with
the alpha-blend model panel = (1 - a) * ct + a * colour; pixels covered by
annotation glyphs (arrows, ID boxes) are filled from the nearest classified pixel.
"""
import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert('RGB'), float) / 255.0


def panels(im, y_probe):
    """Return [(x0, x1, y0, y1)] of framed panels crossing row y_probe.

    A column (row) belongs to the white figure background only if every pixel in
    a wide band is pure white; bright bone never spans the whole band.
    """
    white = im.min(2) > 0.985
    band = white[y_probe - 150:y_probe + 150]
    gapc = band.all(0)
    xs = np.flatnonzero(np.diff(np.r_[1, gapc.astype(int), 1]))
    segs = [(a, b) for a, b in zip(xs[::2], xs[1::2]) if b - a > 200]
    out = []
    for a, b in segs:
        gapr = white[:, a + 40:b - 40].all(1)
        ys = np.flatnonzero(np.diff(np.r_[1, gapr.astype(int), 1]))
        for c, d in zip(ys[::2], ys[1::2]):
            if c <= y_probe < d:
                out.append((a, b, c, d))
    return out


def crop(im, box, inset=6):
    a, b, c, d = box
    return im[c + inset:d - inset, a + inset:b - inset]


def fit_colour(ct, pan, sel):
    """Least-squares alpha-blend fit: pan_ch = (1-a) ct + a c_ch on selected pixels."""
    g = ct[sel].mean(1)
    coefs = [np.polyfit(g, pan[sel][:, ch], 1) for ch in range(3)]
    alpha = 1 - np.mean([c[0] for c in coefs])
    colour = np.array([c[1] for c in coefs]) / max(alpha, 1e-6)
    return alpha, colour


def hue_classes(pan):
    mx, mn = pan.max(2), pan.min(2)
    sat = mx - mn
    r, g, b = pan[..., 0], pan[..., 1], pan[..., 2]
    blue = (sat > 0.15) & (b > r + 0.12) & (b > g + 0.05)
    green = (sat > 0.15) & (g > r + 0.05) & (g > b + 0.05)
    magenta = (sat > 0.15) & (r > g + 0.12) & (b > g + 0.12)
    yellow = (r > 0.75) & (g > 0.6) & (b < 0.45)
    return dict(blue=blue, green=green, magenta=magenta, yellow=yellow)


def classify(ct, pan, models, tol=0.09):
    """Label each pixel 0=background, k=overlay model k, -1=unknown (glyph)."""
    g = ct.mean(2, keepdims=True)
    res = [np.abs(pan - ct).max(2)]
    for a, c in models:
        res.append(np.abs(pan - ((1 - a) * g + a * c)).max(2))
    res = np.stack(res)
    lab = res.argmin(0)
    lab[res.min(0) > tol] = -1
    return lab


def dilate(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def glyphs(ct, pan):
    """Annotation glyphs drawn on top of the overlays: arrows and ID boxes."""
    r, g, b = pan[..., 0], pan[..., 1], pan[..., 2]
    arrow = (r > 0.75) & (g > 0.6) & (b < 0.45)
    box = (pan.max(2) < 0.42) & (b > r + 0.06) & (np.abs(pan - ct).max(2) > 0.08)
    text = (pan.min(2) > 0.85) & (ct.min(2) < 0.75)
    m = arrow | box | text
    return dilate(m, 3)


def fill_unknown(lab, iters=200):
    lab = lab.copy()
    for _ in range(iters):
        unk = lab < 0
        if not unk.any():
            break
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(np.roll(lab, dy, 0), dx, 1)
            take = unk & (sh >= 0)
            lab[take] = sh[take]
            unk = lab < 0
    lab[lab < 0] = 0
    return lab


def majority3(lab, k):
    """3x3 majority filter for small speckle."""
    out = lab.copy()
    pad = np.pad(lab, 1, mode='edge')
    stack = np.stack([pad[1 + dy:1 + dy + lab.shape[0], 1 + dx:1 + dx + lab.shape[1]]
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    for v in range(k + 1):
        cnt = (stack == v).sum(0)
        out[cnt >= 6] = v
    return out


def recover(png, y_probe):
    """Return CT (gray) and class maps for panels [CT, GT, m1, m2, m3]."""
    im = load(png)
    boxes = panels(im, y_probe)
    assert len(boxes) == 5, boxes
    crops = [crop(im, b) for b in boxes]
    h = min(c.shape[0] for c in crops); w = min(c.shape[1] for c in crops)
    crops = [c[:h, :w] for c in crops]
    ct = crops[0]
    # fit one blend model per hue present anywhere among the overlay panels
    models, names = [], []
    for name in ('blue', 'green', 'magenta'):
        sel = np.zeros((h, w), bool); pan_sel = None
        pix_ct, pix_pan = [], []
        for pan in crops[1:]:
            m = hue_classes(pan)[name] & (np.abs(pan - ct).max(2) > 0.1)
            pix_ct.append(ct[m]); pix_pan.append(pan[m])
        pc, pp = np.concatenate(pix_ct), np.concatenate(pix_pan)
        if len(pc) < 200:
            continue
        g = pc.mean(1)
        coefs = [np.polyfit(g, pp[:, ch], 1) for ch in range(3)]
        a = 1 - np.mean([c[0] for c in coefs])
        c = np.array([cc[1] for cc in coefs]) / a
        models.append((a, c)); names.append(name)
    maps = []
    for pan in crops[1:]:
        lab = classify(ct, pan, models)
        lab[glyphs(ct, pan)] = -1
        lab = fill_unknown(lab)
        lab = majority3(lab, len(models))
        maps.append(lab)
    return ct.mean(2), maps, names, models
