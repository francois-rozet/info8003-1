#!/usr/bin/env python

from display import animate
from section1 import *


# 3. Visualization

if __name__ == '__main__':
    h = simulate(stepback, x=(0, 0))

    animate(h, filename='stepback.gif')
