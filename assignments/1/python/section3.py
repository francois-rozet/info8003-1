#!usr/bin/env python

from section2 import *


# 3.a Markov Decision Process

def mdp() -> Tuple[np.array, np.array]:
    '''Markov Decision Process mappings'''

    er = np.zeros((len(U), n, m))  # expected reward
    tp = np.zeros((len(U), n, m, n, m))  # transition probability

    for x in range(n):
        for y in range(m):
            for i, u in enumerate(U):
                for w in W:
                    er[i, x, y] += r((x, y), u, w)
                    tp[i, x, y][f((x, y), u, w)] += 1.

    er /= len(W)
    tp /= len(W)

    return er, tp


# 3.b Q-function

def Q(r: np.array, p: np.array, N: int) -> np.array:
    '''Q-function mapping'''

    q = np.zeros((len(U), n, m))

    for _ in range(N):
        expectation = (p * q.max(axis=0)).sum(axis=(3, 4))
        q = r + gamma * expectation

    return q


def policify(mu: np.array) -> Policy:
    def f(x: State) -> Action:
        return U[mu[x]]

    return f


# 3.c Apply

if __name__ == '__main__':
    decimals = 3
    eps = 10 ** -decimals

    ## Choose N

    N = math.ceil(math.log((eps / (2 * B)) * (1. - gamma) ** 2, gamma))

    print('N =', N)
    print()

    for domain in ['Deterministic', 'Stochastic']:
        printu(f'{domain} domain')
        set_domain(domain.lower())

        ## Compute Q_N

        er, tp = mdp()

        q = Q(er, tp, N)
        q = q.round(decimals)

        print('Q_N(u, x) =', q, sep='\n')
        print()

        ## Compute mu*

        mu_star = q.argmax(axis=0)

        print('mu*(x) =', mu_star, sep='\n')
        print('with', ', '.join(f'{i}: {u}' for i, u in enumerate(U)))
        print()

        ## Compute J^mu*_N

        j = J(policify(mu_star), N)
        j = j.round(decimals)

        print('J^mu*_N(x) =', j, sep='\n')
        print('max_u Q_N(u, x) =', q.max(axis=0), sep='\n')
        print()
