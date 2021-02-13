#!usr/bin/env python

from section3 import *


# 4.a System Identification

def si(h: Trajectory) -> Tuple[np.array, np.array]:
    '''System Identification mappings'''

    c = np.zeros((len(U), n, m), dtype=int)  # occurence count
    r = np.zeros((len(U), n, m))  # expected reward
    p = np.zeros((len(U), n, m, n, m))  # transition probability

    for x0, u0, r0, x1 in h:
        i = U.index(u0)

        c[i][x0] += 1
        r[i][x0] += r0
        p[i][x0][x1] += 1.

    for x in range(n):
        for y in range(n):
            for i, u in enumerate(U):
                if c[i, x, y] == 0:  # if no transition from (u, x, y)
                    p[i, x, y] = 1.  # suppose transitions are equiprobable
                    c[i, x, y] = n * m

    r /= c
    p /= c.reshape(c.shape + (1, 1))

    return r, p


# 4.b Policy

def uniform(x: State) -> Action:
    '''Random uniform policy'''

    return random.choice(U)


# 4.c Apply

if __name__ == '__main__':
    decimals = 3
    eps = 10 ** -decimals

    for domain in ['Deterministic', 'Stochastic']:
        printu(f'{domain} domain')
        set_domain(domain.lower())

        ## Q

        er, tp = mdp()
        N, q = converge(Q(er, tp), eps)

        mu = q.argmax(axis=0)

        def mu_star(x: State) -> Action:
            return U[mu[x]]

        j = J(mu_star, N)
        j = j.round(decimals)

        print('J^mu*_N =', j)
        print()

        ## Q^

        for T in [10 ** i for i in range(6)]:
            print('T =', T)

            ### Compte trajectory

            h = simulate(uniform, (3, 0), T)

            ### Compute r^ and p^

            er_hat, tp_hat = si(h)

            print('||r - r^|| =', np.abs(er_hat - er).max())
            print('||p - p^|| =', np.abs(tp_hat - tp).max())

            ### Compute Q^

            _, q_hat = converge(Q(er_hat, tp_hat), eps)

            print('||Q - Q^|| =', np.abs(q_hat - q).max())
            print()

            ### Compute mû*

            mu_hat = q_hat.argmax(axis=0)

            print('mû*(x) =', mu_hat, sep='\n')
            print('with', ', '.join(f'{i}: {u}' for i, u in enumerate(U)))
            print()

            def mu_hat_star(x: State) -> Action:
                return U[mu_hat[x]]

            ### Compute J^mû_N

            j_hat = J(mu_hat_star, N)
            j_hat = j_hat.round(decimals)

            print('J^mû*_N =', j_hat)
            print()
