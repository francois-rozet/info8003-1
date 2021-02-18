#!usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt


# Settings

plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['savefig.transparent'] = True

if mpl.checkdep_usetex(True):
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.rcParams['text.usetex'] = True


# Classes

class Plot:
    def __init__(self, filename: str = None, **kwargs):
        self.filename = filename
        self.kwargs = kwargs
        self.kwargs.setdefault('figsize', (6, 4))
        self.kwargs.setdefault('tight_layout', True)

    def __enter__(self):
        self.fig = plt.figure(**self.kwargs)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.filename is None:
            plt.show()
        else:
            plt.savefig(self.filename, bbox_inches='tight')
            plt.close()

    def __getattr__(self, attr: str):
        return getattr(plt, attr)
