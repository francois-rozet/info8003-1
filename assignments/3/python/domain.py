#!usr/bin/env python

import random

from typing import Callable, Iterable, List, Tuple


# Types

State = Tuple[float, float]

Action = int  # 0 or 1
Reward = float
Transition = Tuple[State, Action, Reward, State]

Trajectory = Iterable[Transition]
Policy = Callable[[State], Action]


# Globals

U = [-4, 4]

m = 1.
g = 9.81

TIME_STEP = 1e-1
INTEGRATION_STEP = 1e-3

gamma = 0.95

decimals = 2
eps = 1e-2  # precision

B_r = 1  # sup |r|


# Functions

def initial() -> State:
    return random.uniform(-0.1, 0.1), 0


def terminal(x: State) -> bool:
    return abs(x[0]) > 1 or abs(x[1]) > 3


def hill(p: float) -> float:
    if p < 0:
        return p * (p + 1)
    else:
        return p / (1 + 5 * p ** 2) ** (1 / 2)


def hill_prime(p: float) -> float:
    if p < 0:
        return 2 * p + 1
    else:
        return 1 / (1 + 5 * p ** 2) ** (3 / 2)


def hill_second(p: float) -> float:
    if p < 0:
        return 2
    else:
        return -15 * p / (1 + 5 * p ** 2) ** (5 / 2)


def dynamics(x: State, u: Action) -> State:
    '''Environment dynamics'''

    p, s = x

    for _ in range(int(TIME_STEP // INTEGRATION_STEP)):
        h1 = hill_prime(p)
        h2 = hill_second(p)

        p_prime = s
        s_prime = (U[u] / m - g * h1 - s ** 2 * h1 * h2) / (1 + h1 ** 2)

        p = p + INTEGRATION_STEP * p_prime
        s = s + INTEGRATION_STEP * s_prime

    return p, s


def reward(x: State, u: Action) -> Tuple[Reward, State]:
    '''Reward signal'''

    p, s = dynamics(x, u)

    if p < -1 or abs(s) > 3:
        r = -1
    elif p > 1:
        r = 1
    else:
        r = 0

    return r, (p, s)


def onestep(x: State, u: Action) -> Transition:
    '''One-step transition simulator'''

    return (x, u) + reward(x, u)


def simulate(mu: Policy, x: State = None) -> Trajectory:
    '''Trajectory simulator'''

    if x is None:
        x = initial()

    while not terminal(x):
        u = mu(x)
        l = onestep(x, u)

        yield l

        x = l[-1]
