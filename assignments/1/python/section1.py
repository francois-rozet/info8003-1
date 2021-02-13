#!usr/bin/env python

import math
import numpy as np
import random

from typing import Callable, Iterable, List, Tuple


# 0. Helpers

## Miscellaneous

def printu(*args, **kwargs):
    '''Print with underline'''

    print(*args, **kwargs)
    print(*map(lambda t: '-' * len(t), args), **kwargs)


## Types

State = Tuple[int, int]  # (x, y)
Action = Tuple[int, int]  # (i, j)
Reward = float  # r
Noise = float  # w

Transition = Tuple[State, Action, Reward, State]  # (x_i, u_i, r_i, x_{i+1})
Trajectory = List[Transition]
Policy = Callable[[State], Action]  # mu: X -> U


# 1.a Domain

## Global variables

n, m = 5, 5

g = np.array([
    [ -3.,   1., -5.,  0., 19.],
    [  6.,   3.,  8.,  9., 10.],
    [  5.,  -8.,  4.,  1., -8.],
    [  6.,  -9.,  4., 19., -5.],
    [-20., -17., -4., -3.,  9.]
])

B = g.max()

U = [(1, 0), (-1, 0), (0, 1), (0, -1)]

gamma = 0.99

W = [0.]


## Components

def F(x: State, u: Action) -> State:
    x, y = x
    i, j = u

    return min(max(x + i, 0), n - 1), min(max(y + j, 0), m - 1)


def f(x: State, u: Action, w: Noise) -> State:
    '''Dynamics function'''

    if w <= 0.5:
        return F(x, u)

    return (0, 0)


def R(g: np.array, x: State) -> Reward:
    return g[x]


def r(x: State, u: Action, w: Noise) -> Reward:
    '''Reward signal function'''

    return R(g, f(x, u, w))


def w(x: State, u: Action) -> Noise:
    '''Noise sampler'''

    return random.choice(W)


def set_domain(mode: str):
    global W
    W.clear()

    if mode == 'stochastic':
        W += [0., 1.]  # U{0., 1.} replaces U(0., 1.)
    else:
        W += [0.]


# 1.b Rule-based policy

def clockwise(x: State) -> Action:
    '''Clockwise-lap stationary policy'''

    x, y = x

    if x < n - 1 and y == m - 1:
        return (1, 0)
    elif x == n - 1 and y > 0:
        return (0, -1)
    elif x > 0 and y == 0:
        return (-1, 0)
    else:
        return (0, 1)


def simulate(mu: Policy, x0: State, T: int) -> Trajectory:
    '''Trajectory simulator'''

    h = []

    for t in range(T):
        u0 = mu(x0)
        w0 = w(x0, u0)
        r0 = r(x0, u0, w0)
        x1 = f(x0, u0, w0)

        h.append((x0, u0, r0, x1))

        x0 = x1

    return h


if __name__ == '__main__':
    for domain in ['Deterministic', 'Stochastic']:
        printu(f'{domain} domain')
        set_domain(domain.lower())

        h = simulate(clockwise, (3, 0), 11)

        print(*h, sep='\n')
        print()
