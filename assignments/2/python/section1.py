#!/usr/bin/env python

from domain import *


# 1.a Rule-based policies

def uniform(x: State) -> Action:
    '''Random uniform policy'''

    return random.choice(U)


def notsofastandfurious(x: State) -> Action:
    '''Full-throttle policy'''

    return U[1]


def stepback(x: State) -> Action:
    '''Sometimes-you-have-to-take-a-step-back-to-move-forward policy'''

    if -0.5 < x[0] < 0 and -1.5 < x[1] < 0:
        return U[0]
    else:
        return U[1]


# 1.b Apply

if __name__ == '__main__':
    h = simulate(stepback)

    print(*h, sep='\n')
