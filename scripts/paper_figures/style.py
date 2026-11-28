"""Shared figure style for the manuscript (IEEE two-column, Times text)."""
import matplotlib as mpl

TEXTWIDTH = 7.16   # in, IEEE conference \textwidth
COLWIDTH = 3.5     # in, IEEE conference \columnwidth

# Score identity colours (Okabe-Ito based; validated for CVD separation).
C_COV = '#0072B2'      # coverage A
C_GRP = '#D55E00'      # grouping wQ_vox
C_GRPJ = '#E69F00'     # grouping wQ_joint
C_DVOX = '#7A5195'     # combined D_vox
C_DJNT = '#009E73'     # combined D_joint
C_ENT = '#8C8C8C'      # voxel entropy
C_RAND = '#222222'     # random ranking
INK = '#222222'
MUTED = '#6B6B6B'
GRID = '#E6E6E6'

# Fragment overlay colours (identity of fragments only).
FRAG = ['#56B4E9', '#B39DDB', '#F28CB1', '#7FD1AE']


def apply():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'font.size': 7.5,
        'axes.titlesize': 8,
        'axes.labelsize': 7.5,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6.8,
        'axes.linewidth': 0.6,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 600,
        'figure.dpi': 150,
    })


def panel_label(ax, letter, title, x=0.0, y=1.0, **kw):
    """Bold panel letter followed by a short title, left-aligned above the axes."""
    ax.text(x, y, f'$\\bf{{{letter}}}$  {title}', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8, color=INK, **kw)
