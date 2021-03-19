#!/usr/bin/env python

import numpy as np
import tqdm

from section2 import *


# 4.a Estimators

## Linear Regression

def LR(**kwargs):
    from sklearn.linear_model import LinearRegression

    return LinearRegression(**kwargs)


## Extremely Randomized Trees

def XRT(**kwargs):
    from sklearn.ensemble import ExtraTreesRegressor

    kwargs.setdefault('n_estimators', 20)

    return ExtraTreesRegressor(**kwargs)


## Neural Networks

def KMLP(**kwargs):
    import tensorflow as tf
    import tensorflow.keras as ass

    class MLP(ass.Sequential):
        """Keras Multi-Layer Perceptron"""

        def __init__(
            self,
            input_size: int = 3,
            output_size: int = 1,
            hidden_size: int = 8,
            n_layers: int = 3,
            activation: str = 'relu',
            batch_size: int = 32,
            epochs: int = 5,
            seed: int = 0
        ):
            super().__init__()

            np.random.seed(seed)
            tf.random.set_seed(seed)

            self.add(ass.Input(shape=(input_size,)))

            for _ in range(n_layers):
                self.add(ass.layers.Dense(hidden_size, activation=activation))

            self.add(ass.layers.Dense(output_size))

            self.compile(
                optimizer='adam',
                loss='mse'
            )

            self.batch_size = batch_size
            self.epochs = epochs

        def fit(self, *args, **kwargs):
            kwargs.setdefault('batch_size', self.batch_size)
            kwargs.setdefault('epochs', self.epochs)
            kwargs.setdefault('verbose', False)
            super().fit(*args, **kwargs)

    return MLP(**kwargs)


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

def fqi(model, ts: TrainingSet, N: int = None, threshold: float = 0.02):
    '''Fitted-Q-Iteration training'''

    stateaction, reward, state_prime = ts

    stateaction_prime = np.hstack((
        np.repeat(state_prime, len(U), axis=0),
        np.tile(U, len(state_prime))[:, None]
    ))

    terminal = reward != 0
    expected_return = reward

    prev_q = None

    for _ in tqdm.tqdm(count(1) if N is None else range(1, N + 1)):
        model.fit(stateaction, expected_return)

        q = model.predict(stateaction_prime)
        max_q = q.reshape(-1, len(U)).max(axis=1)

        expected_return = np.where(terminal, reward, gamma * max_q)

        if N is None:
            if prev_q is not None:
                mae = np.abs(q - prev_q).mean()
                if mae < threshold:
                    break

            prev_q = q


## 4.d Expected return of policy

def policify(mu: np.array) -> Policy:
    mu *= 4

    def f(x: State) -> Action:
        p, s = x
        p, s = (p + 1) / 2, (s + 3) / 6
        i, j = int(p * mu.shape[0]), int(s * mu.shape[1])
        i, j = min(i, mu.shape[0] - 1), min(j, mu.shape[1] - 1)

        return mu[i, j]

    return f


# 4.e Apply

if __name__ == '__main__':
    from plots import plt

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

        print(generator.__name__)
        print('-' * len(generator.__name__))
        print()

        for stop in [1, 2]:

            print(f'stopping rule {stop}')
            print('.' * 15)
            print()

            if stop == 1:
                N = math.ceil(math.log((eps / (2 * B_r)) * (1. - gamma), gamma))
            elif stop == 2:
                N = None

            for method in [LR, XRT, KMLP]:
                model = method()

                print(method.__name__)

                fqi(model, ts, N)

                ### Compute Q^_N

                qq = model.predict(stateaction).reshape(gridshape)

                ### Compute mû_N

                mu_hat = 2 * qq.argmax(axis=-1) - 1

                ### Plots

                for key, zz in {'q_-4': qq[..., 0], 'q_+4': qq[..., 1], 'mu': mu_hat}.items():

                    plt.pcolormesh(
                        p, s, zz.T,
                        cmap='coolwarm_r',
                        vmin=-1, vmax=1,
                        rasterized=True
                    )
                    plt.xlabel(r'$p$')
                    plt.ylabel(r'$s$')

                    if 'q' in key:
                        plt.colorbar()

                    plt.savefig(f'4_{generator.__name__}_{stop}_{method.__name__}_{key}.pdf')
                    plt.close()

                ### Compute J^mû_N'

                N_prime = math.ceil(math.log((eps / B_r), gamma))

                trajectories = samples(policify(mu_hat), N_prime)
                j_hat = expected_return(trajectories, N_prime)

                print('J^mû_N =', j_hat)
                print()
