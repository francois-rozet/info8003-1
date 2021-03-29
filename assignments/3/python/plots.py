#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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


# Plot

def plot(input_file: str, output_file: str):
    ## State space

    resolution = 0.01

    p = np.arange(-1., 1., resolution) + resolution / 2
    s = np.arange(-3., 3., resolution) + resolution / 2

    ## Data

    data = np.loadtxt(input_file)

    ## Plot

    plt.pcolormesh(
        p, s, data.T,
        cmap='coolwarm_r',
        vmin=-1, vmax=1,
        shading='auto',
        rasterized=True
    )
    plt.xlabel(r'$p$')
    plt.ylabel(r'$s$')

    plt.colorbar()

    plt.savefig(output_file)
    plt.close()


# Main

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot Q-function')

    parser.add_argument('input', help='input TXT file')
    parser.add_argument('output', help='output PDF file')

    args = parser.parse_args()

    plot(
    	input_file=args.input,
    	output_file=args.output
    )
