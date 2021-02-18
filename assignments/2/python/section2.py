#!usr/bin/env python

import math

from itertools import islice

from plots import Plot
from section1 import *


# 2.a Expected Return of a Policy

def nth(h: Trajectory, N: int) -> Tuple[int, Transition]:
    '''N-th transition or last'''

    if type(h) is list:
        n = min(len(h), N) - 1
        l = h[n]
    else:
        for n, l in zip(range(N), h):
            pass

    return n, l


def cumulative_reward(h: Trajectory, N: int) -> Transition:
    '''Cumulative reward after N steps'''

    n, (_, _, r, _) = nth(h, N)

    return (gamma ** n) * r  # taking advantage of null rewards


def expected_return(trajectories: List[Trajectory], N: int) -> Reward:
    '''Expected return by Monte Carlo'''

    total = sum(cumulative_reward(h, N) for h in trajectories)

    return total / len(trajectories)


# 2.b Apply

if __name__ == '__main__':

    ## Choose N

    N = math.ceil(math.log((eps / B_r), gamma))

    print('N =', N)

    ## Compute J^mu_N

    ### Simulate 50 trajectories

    trajectories = []

    for _ in range(50):
        h = list(islice(simulate(stepback), N))  # truncated
        trajectories.append(h)

    ### Plot

    N = list(range(1, N))
    J = []

    for n in N:
        J.append(expected_return(trajectories, n))

    with Plot('2_expected_return.pdf') as plt:
        plt.plot(N, J)
        plt.xlabel(r'$N$')
        plt.ylabel(r'$J^\mu_N$')
