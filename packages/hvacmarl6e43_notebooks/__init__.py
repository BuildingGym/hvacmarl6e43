from hvacmarl6e43.utils import resolve_path

notebook_storage_base = resolve_path('__datastore__')

import matplotlib.pyplot as plt

# TODO
def setup():
    from pathlib import Path
    Path(notebook_storage_base).mkdir(exist_ok=True)

    plt.rcParams.update({
        'backend': 'svg',
        'pgf.texsystem': 'lualatex',
        'pgf.rcfonts': False,
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': [],
    })


# TODO plt.style.available
# plt.style.use('seaborn-v0_8-paper')

import matplotlib.axis as maxis
import matplotlib.dates as mdates

def mpl_autofmt_datetime(axis: maxis.Axis):
    axis.set_major_locator(locator := mdates.AutoDateLocator())
    axis.set_major_formatter(formatter := mdates.ConciseDateFormatter(axis.get_major_locator()))
    return locator, formatter

from matplotlib.patches import Patch

def mpl_shared_legend_patches(axes):
    return [
        Patch(color=color, label=label) 
        for (label, color) in dict.fromkeys([
            (label, handle.get_color())
            for ax in axes
            for handle, label in zip(*ax.get_legend_handles_labels())
        ])
    ]