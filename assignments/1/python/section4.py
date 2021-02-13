#!usr/bin/env python

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

def norm(x: np.array) -> float:
    '''Infinite norm'''

    return np.abs(x).max()


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

        ## Q

        er, tp = mdp()
        q = Q(er, tp, N)

        mu = q.argmax(axis=0)

        def mu_star(x: State) -> Action:
            return U[mu[x]]

        j = J(mu_star, N)
        j = j.round(decimals)

        print('J^mu*_N =', j)
        print()

        ## Q^

        r_norm = []
        p_norm = []
        q_norm = []

        ### Simulate longest trajectory

        T = [10 ** i for i in range(7)]
        h = simulate(uniform, (3, 0), T[-1])

        for t in T:
            print('T =', t)

            ### Compute r^, p^ and Q^

            er_hat, tp_hat = si(h[:t])  # truncated trajectory
            q_hat = Q(er_hat, tp_hat, N)

            r_norm.append(norm(er_hat - er))
            p_norm.append(norm(tp_hat - tp))
            q_norm.append(norm(q_hat - q))

            print('||r - r^|| =', r_norm[-1])
            print('||p - p^|| =', p_norm[-1])
            print('||Q - Q^|| =', q_norm[-1])
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

        ## Plots

        fig = plt.figure(figsize=(6, 4))
        plt.plot(T, p_norm, '--o')
        plt.xscale('log')
        plt.xlabel('Trajectory length')
        plt.ylabel(r'$\left\| \hat{p} - p \right\|$')
        plt.grid()
        plt.savefig(f'{domain.lower()}_p_norm.pdf', **flags)
        plt.close()

        fig = plt.figure(figsize=(6, 4))
        plt.plot(T, r_norm, '--o')
        plt.xscale('log')
        plt.xlabel('Trajectory length')
        plt.ylabel(r'$\left\| \hat{r} - r \right\|$')
        plt.grid()
        plt.savefig(f'{domain.lower()}_r_norm.pdf', **flags)
        plt.close()
