#!/usr/bin/env python3

"""
Implementation of Fitted-Q-Iteration with Extremely Randomized Trees (FQI with
XRT) algorithm for the Double Inverted Pendulum control problem.
"""

###########
# Imports #
###########

import gym
import math
import numpy as np
import pybulletgym

from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm
from typing import Tuple

from plots import plt


############
# Settings #
############

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


##########
# Typing #
##########

State = np.array  # (9)
Action = float
Reward = float
Done = bool

Transition = Tuple[State, Action, Reward, State, Done]


###########
# Classes #
###########

class FQI:
    '''Fitted-Q-Iteration agent'''

    def __init__(
        self,
        U: np.array,
        U_dim: int,
        gamma: float = 0.95,
        n_estimators: int = 20
    ):
        # Discrete action space and its dimension
        self.U = U
        self.U_dim = U_dim

        # Model used with FQI algorithm
        self.model = ExtraTreesRegressor(n_estimators=n_estimators)

        # List of transitions stored
        self.memory = []

        # Saved target and discount factor
        self.target = None
        self.gamma = gamma

    def fill(self, env, n: int = 10000, max_steps: int = 1000) -> None:
        '''Fill the memory with `n` transitions'''

        for i in range(n):
            # Reset the environment after `max_steps` steps
            if not i % max_steps:
                x = env.reset()

            # Generate action with random policy
            u = env.action_space.sample()

            # Play action and store transition
            x_prime, r, done, _ = env.step(u)
            self.memory.append((x, u[0], r, x_prime, done))

            if done:
                x = env.reset()

    def action(self, x: State) -> np.array:
        '''Get an action based on trained model prediction'''

        # State-action
        xu = np.concatenate((
            np.tile(x, (len(self.U), 1)),
            self.U.reshape(-1, self.U_dim)
        ), axis=1)

        # Predict Q and return best action
        q = self.model.predict(xu).reshape(-1, len(self.U))

        return self.U[np.argmax(q, axis=-1)].reshape(1)

    def optimize(self) -> None:
        '''Train the model using all transitions stored'''

        # Get all transitions stored
        x, u, r, x_prime, done = map(np.array, zip(*self.memory))

        # State-actiona and state-action prime
        xu = np.hstack((x, u[:, None]))
        xu_prime = np.hstack((
            np.repeat(x_prime, len(self.U), axis=0),
            np.tile(self.U, len(x_prime))[:, None]
        ))

        # Compute the target
        if self.target is None:
            self.target = r
        else:
            q = self.model.predict(xu_prime)
            max_q = q.reshape(-1, len(self.U)).max(axis=1)

            self.target = r + self.gamma + (1 - done) * max_q

        # Train the model
        self.model.fit(xu, self.target)


########
# Main #
########

def main(
    render: bool = False,
    eps: float = 1.0,
    n_actions: int = 11,
    n_estimators: int = 20
):
    # Environment

    gym.logger.set_level(40)
    env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')

    # Rendering

    if render:
        env.render()
        env.reset()

        # /!\ Only work in our modified version of PyBullet Gym
        env.camera.env._p.resetDebugVisualizerCamera(2, 0, -20, [0, 0, 0])

    # Setup

    gamma = 0.95
    max_steps = 1000
    n_evaluate = 20  # number of episodes to evaluate model

    B_r = 10  # maximum possible reward
    N = math.ceil(math.log((eps / (2 * B_r)) * (1. - gamma) ** 2, gamma))

    print(f'N = {N}')

    rewards = []

    # Discrete actions

    actions = np.linspace(
        env.action_space.low[0],
        env.action_space.high[0],
        n_actions
    )

    # Agent

    agent = FQI(
        U=actions,
        U_dim=env.action_space.shape[0],
        gamma=gamma,
        n_estimators=n_estimators
    )
    agent.fill(env, 3000, max_steps)  # generate 3000 transitions to start

    # Training

    for _ in tqdm(range(N)):
        # Train the model

        agent.optimize()

        # Evaluate

        evals = []

        for _ in range(n_evaluate):
            x = env.reset()
            cr = 0  # cumulative reward

            for step in range(max_steps):
                u = agent.action(x)
                x_prime, r, done, _ = env.step(u)

                # Store the transition in agent's transitions list
                agent.memory.append((x, u[0], r, x_prime, done))

                x = x_prime

                if done:
                    break

                cr += (gamma ** step) * r

            evals.append(cr)

        print(f'Memory size: {len(agent.memory)}')

        rewards.append(evals)

    # Export results

    rewards = np.array(rewards)

    mean = np.mean(rewards, axis=1)
    std = np.std(rewards, axis=1)

    plt.plot(mean)
    plt.fill_between(
        range(N),
        mean - std,
        mean + std,
        alpha=0.3
    )

    plt.xlabel('N')
    plt.ylabel(r'$J^{\mu}$')

    plt.savefig(f'fqi_J_{n_actions}_{n_estimators}.pdf')
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-render', default=False, action='store_true')
    parser.add_argument('-eps', type=float, default=1.0)
    parser.add_argument('-actions', type=int, default=11)
    parser.add_argument('-estimators', type=int, default=20)

    args = parser.parse_args()

    main(
        render=args.render,
        eps=args.eps,
        n_actions=args.actions,
        n_estimators=args.estimators
    )
