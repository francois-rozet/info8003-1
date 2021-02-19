#!usr/bin/env python

import math
import numpy as np
import tensorflow.keras as ass

from itertools import islice
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

from plots import Plot
from section1 import *


# 4.a Estimators

## Linear Regression

class LR(LinearRegression):
    pass


## Extremely Randomized Trees

class XRT(ExtraTreesRegressor):
    def __init__(self, **kwargs):
        kwargs.setdefault('n_estimators', 20)
        super().__init__(**kwargs)


## Neural Networks

class MLP(ass.Sequential):
    def __init__(
        self,
        input_size: int = 3,
        output_size: int = 1,
        hidden_size: int = 8,
        n_layers: int = 3,
        activation: str = 'relu',
        epochs: int = 5
    ):
        super().__init__()

        self.add(ass.Input(shape=(input_size,)))

        for _ in range(n_layers):
            self.add(ass.layers.Dense(hidden_size, activation=activation))

        self.add(ass.layers.Dense(output_size))

        self.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        self.epochs = epochs

    def fit(self, *args, **kwargs):
        kwargs.setdefault('epochs', self.epochs)
        super().fit(*args, **kwargs)


# 4.b Generation of the training-set

TrainingSet = Tuple[np.array, np.array, np.array]

def training_set(h: Trajectory) -> TrainingSet:
    '''Trajectory to training-set'''

    x, u, r, x_prime = map(np.array, zip(*h))

    xu = np.hstack((x, u[:, None]))

    return xu, r, x_prime


def exhaustive(steps: int = 200) -> Trajectory:
    '''Exhaustive grid generator'''

    h = []

    for p in np.linspace(-1, 1, steps):
        for s in np.linspace(-3, 3, steps):
            for u in U:
                h.append(onestep((p, s), u))

    return h


def montecarlo(n: int = 80000, N: int = 1000, seed: int = 0):
    '''Monte Carlo random generator'''

    random.seed(seed)

    h = []

    while len(h) < n:
        h.extend(islice(simulate(uniform), N))

    return h[:n]


# 4.c Fitted-Q-Iteration

def fqi(model, ts: TrainingSet, N: int):
    '''Fitted-Q-Iteration training'''

    stateaction, reward, state_prime = ts

    stateaction_prime = np.hstack((
        np.repeat(state_prime, len(U), axis=0),
        np.tile(U, len(state_prime))[:, None]
    ))

    terminal = reward != 0
    expected_return = reward

    for _ in range(N):
        model.fit(stateaction, expected_return)

        q = model.predict(stateaction_prime)
        max_q = q.reshape(-1, len(U)).max(axis=1)

        expected_return = np.where(terminal, reward, gamma * max_q)


# 4.d Apply

if __name__ == '__main__':

    ## Choose N

    N = math.ceil(math.log((eps / (2 * B_r)) * (1. - gamma), gamma))

    print('N =', N)

    ## Mesh

    p = np.linspace(-1, 1, 200)
    s = np.linspace(-3, 3, 600)

    pp, ss, uu = np.meshgrid(p[:-1], s[:-1], U, indexing='ij')
    stateaction = np.vstack((pp.ravel(), ss.ravel(), uu.ravel())).T

    gridshape = pp.shape

    del pp, ss, uu


    ## FIQ

    for generator in [exhaustive, montecarlo]:
        ts = training_set(generator())

        for method in [LR, XRT, MLP]:
            model = method()

            fqi(model, ts, N)

            qq = model.predict(stateaction).reshape(gridshape)
            mu = 2 * qq.argmax(axis=-1) - 1

            for key, zz in {'q_-4': qq[..., 0], 'q_+4': qq[..., 1], 'mu': mu}.items():
                with Plot(f'4_{key}_{generator.__name__}_{method.__name__}.pdf') as plt:
                    plt.pcolormesh(p, s, zz.T, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.xlabel(r'$p$')
                    plt.ylabel(r'$s$')

                    if 'q' in key:
                        plt.colorbar()
