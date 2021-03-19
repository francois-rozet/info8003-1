#!/usr/bin/env python

from section1 import *


# 2.a Expected Return

def J(mu: Policy, N: int) -> np.array:
    '''Expected return mapping'''

    j = np.zeros((n, m))

    for _ in range(N):
        temp = np.zeros(j.shape)

        for x in range(n):
            for y in range(m):
                x0 = (x, y)
                u0 = mu(x0)

                for w0 in W:
                    r0 = r(x0, u0, w0)
                    x1 = f(x0, u0, w0)

                    temp[x0] += r0 + gamma * j[x1]

        j = temp / len(W)

    return j


# 2.b Apply

if __name__ == '__main__':
    decimals = 3
    eps = 10 ** -decimals

    ## Choose N

    N = math.ceil(math.log((eps / B) * (1. - gamma), gamma))

    print('N =', N)
    print()

    for domain in ['Deterministic', 'Stochastic']:
        printu(f'{domain} domain')
        set_domain(domain.lower())

        ## Compute J^mu_N

        j = J(clockwise, N)
        j = j.round(decimals)

        print('J^mu_N(x) =', j, sep='\n')
        print()
