#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt


# Settings

plt.rcParams['axes.grid'] = True
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 14.
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['savefig.transparent'] = True

if mpl.checkdep_usetex(True):
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.rcParams['text.usetex'] = True
