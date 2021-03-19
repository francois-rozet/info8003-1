#!/usr/bin/env python

import matplotlib.pyplot as plt

from matplotlib import rc

from section3 import *


# 0. Settings

plt.rcParams['font.size'] = 16

flags = {
    'transparent': True,
    'bbox_inches': 'tight'
}

## /!\ Requires LaTeX

# rc('font', **{'family': ['serif'], 'serif': ['Computer Modern']})
# rc('text', usetex=True)


# 4.a System Identification

def si(h: Trajectory) -> Tuple[np.array, np.array]:
    '''System Identification mappings'''

    # Count

    N = np.zeros((len(U), n, m, n, m))  # action-state-state count
    R = np.zeros((len(U), n, m))  # total reward

    for x0, u0, r0, x1 in h:
        i = U.index(u0)

        N[i][x0 + x1] += 1.
        R[i][x0] += r0

    # Correction

    miss = N.sum(axis=(-1, -2)) == 0.  # if no occurence of (u, x)
    N[miss] = 1.  # suppose all transitions from (u, x) are equiprobable

    # Average

    M = N.sum(axis=(-1, -2), keepdims=True)  # action-state count

    r = R / M.squeeze()  # expected reward
    p = N / M  # transition probability

    return r, p


# 4.b Policy

def uniform(x: State) -> Action:
    '''Random uniform policy'''

    return random.choice(U)


# 4.c Apply

def norm(x: np.array) -> float:
    '''Infinite norm'''

    return np.abs(x).max()


if __name__ == '__main__':
    decimals = 3
    eps = 10 ** -decimals

    N = math.ceil(math.log((eps / (2 * B)) * (1. - gamma) ** 2, gamma))

    for domain in ['Deterministic', 'Stochastic']:
        printu(f'{domain} domain')
        set_domain(domain.lower())

        ## Compute Q, mu* and J^mu*

        er, tp = mdp()
        q = Q(er, tp, N)

        mu_star = q.argmax(axis=0)

        j = J(policify(mu_star), N)

        ## Simulate trajectory

        T = [10 ** i for i in range(7)]
        h = simulate(uniform, (3, 0), T[-1], seed=2)

        ## Compute r^, p^ and Q^

        r_norm = []
        p_norm = []
        q_norm = []

        for t in T:
            er_hat, tp_hat = si(h[:t])  # truncated trajectory
            q_hat = Q(er_hat, tp_hat, N)

            r_norm.append(norm(er_hat - er))
            p_norm.append(norm(tp_hat - tp))
            q_norm.append(norm(q_hat - q))

        ## Compute mû*

        mu_hat_star = q_hat.argmax(axis=0)

        print('mû*(x) =', mu_hat_star, sep='\n')
        print()

        ## Compute J^mû_N

        j_hat = J(policify(mu_hat_star), N)
        j_hat = j_hat.round(decimals)

        print('J^mû*_N(x) =', j_hat, sep='\n')
        print()

        ## Plots

        for x, x_norm in {'r': r_norm, 'p': p_norm, 'Q': q_norm}.items():
            fig = plt.figure(figsize=(6, 4))
            plt.plot(T, x_norm, '--o', markersize=5)
            plt.xscale('log')
            plt.xlabel('$t$')
            plt.ylabel(f'$\\left\\| \\hat{{{x}}} - {x} \\right\\|_\\infty$')
            plt.grid()
            plt.savefig(f'4_{x}_{domain.lower()}.pdf', **flags)
            plt.close()
