#!usr/bin/env python

from section4 import *


# 5.1 Offline Q-Learning

def update(q: np.array, l: Transition, alpha: float) -> None:
    '''In-place Q-learning update'''

    x0, u0, r0, x1 = l
    i = U.index(u0)

    delta = r0 + gamma * q[(...,) + x1].max() - q[i][x0]
    q[i][x0] += alpha * delta


def offline(h: Trajectory, alpha: float = 0.05) -> np.array:
    '''Offline Q-learning protocol'''

    q = np.zeros((len(U), n, m))

    for l in h:
        update(q, l, alpha)

    return q


# 5.2 Online Q-Learning

QPolicy = Callable[[State, np.array], Action]

def greedy(x: State, q: np.array, epsilon: float = 0.25) -> Action:
    '''Epsilon-greedy policy'''

    if random.random() < epsilon:
        return uniform(x)
    else:
        return U[q[(...,) + x].argmax()]


def online(
    qmu: QPolicy,
    x0: State,
    T: int = 1000,
    epochs: int = 100,
    alpha: float = 0.05,
    decay: float = 1.,
    k: int = 0,
    clear: bool = False
) -> Iterable[np.array]:
    '''Online Q-learning protocol'''

    q = np.zeros((len(U), n, m))

    h = []

    for _ in range(epochs):
        x = x0

        for t in range(T):
            u = qmu(x, q)
            l = step(x, u)
            h.append(l)

            update(q, l, alpha)
            for l in random.choices(h, k=k):
                update(q, l, alpha)

            alpha *= decay

            x = l[-1]

        if clear:
            h.clear()

        yield q


# 5.3 Discount Factor

# gamma = 0.4 in section1.py


# 5.4 Q-Learning with Another Exploration Policy

tau = (1. - gamma) / B

def boltzmann(x: State, q: np.array) -> Action:
    '''Boltzmann exploration policy'''

    p = np.exp(q[(...,) + x] * tau)
    p /= p.sum()

    return U[np.random.choice(len(U), p=p)]


# 5. Apply

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

        ## 5.1

        ### Simulate trajectory

        T = [10 ** i for i in range(7)]
        h = simulate(uniform, (3, 0), T[-1], seed=2)

        ### Compute Q^

        q_norm = []

        for t in T:
            q_hat = offline(h[:t])  # truncated trajectory
            q_norm.append(norm(q_hat - q))

        fig = plt.figure(figsize=(6, 4))
        plt.plot(T, q_norm, '--o', markersize=5)
        plt.xscale('log')
        plt.xlabel('$t$')
        plt.ylabel(f'$\\left\\| \\hat{{Q_t}} - Q \\right\\|_\\infty$')
        plt.grid()
        plt.savefig(f'5.1_{domain.lower()}.pdf', **flags)
        plt.close()

        ### Compute mû* and J^mû*

        mu_hat_star = q_hat.argmax(axis=0)

        print('mû*(x) =', mu_hat_star, sep='\n')
        print()

        j_hat = J(policify(mu_hat_star), N)
        j_hat = j_hat.round(decimals)

        print('J^mû*_N(x) =', j_hat, sep='\n')
        print()

        ## 5.2 & 5.4

        protocols = {
            '5.2.1': online(greedy, (3, 0)),
            '5.2.2': online(greedy, (3, 0), decay=0.8),
            '5.2.3': online(greedy, (3, 0), k=10),
            '5.2.3bis': online(greedy, (3, 0), k=10, clear=True),
            '5.4': online(boltzmann, (3, 0))
        }

        for key, protocol in protocols.items():
            temp = []

            for q_hat in protocol:
                mu_hat = q_hat.argmax(axis=0)  # mu^(x)
                q_mu_hat = q_hat.max(axis=0)  # Q^(x, mu^(x))
                j_hat = J(policify(mu_hat), N)  #J^mu^_N(x)

                temp.append((
                    norm(q_mu_hat - j),
                    j[3, 0],
                    q_mu_hat[3, 0],
                    j_hat[3, 0]
                ))

            temp = list(zip(*temp))

            fig = plt.figure(figsize=(6, 4))
            plt.plot(temp[0], label='$\\left\\| \\hat{{Q}} - J^{{\\mu^*}}_N \\right\\|_\\infty$')
            plt.plot(temp[1], label='$J^{{\\mu^*}}_N(3, 0)$')
            plt.plot(temp[2], label='$\\hat{{Q}}((3, 0), \\hat{{\\mu}}(3, 0))$')
            plt.plot(temp[3], label='$J^{{\\hat{{\\mu}}}}_N(3, 0)$')
            plt.xlabel('Episode')
            plt.grid()
            plt.legend(prop={'size': 8})
            plt.savefig(f'{key}_{domain.lower()}.pdf', **flags)
            plt.close()
