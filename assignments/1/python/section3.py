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

def Q(r: np.array, p: np.array) -> Iterable[np.array]:
    '''Q-function mapping sequence'''

    q = np.zeros((len(U), n, m))
    yield q

    while True:
        expectation = (p * q.max(axis=0)).sum(axis=(3, 4))
        q = r + gamma * expectation
        yield q


def converge(seq: Iterable[np.array], eps: float) -> Tuple[int, np.array]:
    for i, curr in enumerate(seq):
        if i > 0:
            if np.abs(curr - prev).max() < eps:
                return i - 1, prev

        prev = curr


# 3.c Apply

if __name__ == '__main__':
    decimals = 3
    eps = 10 ** -decimals

    for domain in ['Deterministic', 'Stochastic']:
        printu(f'{domain} domain')
        set_domain(domain.lower())

        ## Compute N and Q_N

        er, tp = mdp()

        N, q = converge(Q(er, tp), eps)
        q = q.round(decimals)

        print('N =', N)
        print('Q_N(u, x) =', q, sep='\n')
        print()

        ## Compute mu*

        mu = q.argmax(axis=0)

        print('mu*(x) =', mu, sep='\n')
        print('with', ', '.join(f'{i}: {u}' for i, u in enumerate(U)))
        print()

        def mu_star(x: State) -> Action:
            return U[mu[x]]

        ## Compute J^mu*_N

        j = J(mu_star, N)
        j = j.round(decimals)

        print('J^mu*_N(x) =', j, sep='\n')
        print('max_u Q_N(u, x) =', q.max(axis=0), sep='\n')
        print()
