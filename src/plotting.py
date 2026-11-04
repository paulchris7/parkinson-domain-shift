"""Consistent figure style for the paper and the poster."""
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as C


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
    })


def save(fig, name):
    """Write the figure as PNG and PDF, show it in the notebook, then free it."""
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"{name}.{ext}")
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        pass
    plt.close(fig)
