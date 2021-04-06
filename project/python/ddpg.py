#!/usr/bin/env python3

"""
Implementation of Deep Deterministic Policy Gradient (DDPG) algorithm for the
Double Inverted Pendulum control problem.

Sources:
    [1] Continuous control with deep reinforcement learning,
        Lillicrap et al.,
        https://arxiv.org/abs/1509.02971
    [2] Deep Reinforcement Learning in Large Discrete Action Spaces,
        Dulac-Arnold et al.
        https://arxiv.org/abs/1512.07679
"""

###########
# Imports #
###########

import gym
import numpy as np
import pybulletgym
import random
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from typing import Tuple

from plots import plt


##########
# Typing #
##########

State = torch.Tensor  # (9)
Action = float  # [-1, 1]
Reward = float
Done = bool

Transition = Tuple[State, Action, Reward, State, Done]

Batch = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]


###########
# Classes #
###########

# Replay buffer

class ReplayBuffer:
    '''Cyclic buffer of bounded capacity'''

    def __init__(self, capacity: int = 2 ** 16, batch_size: int = 32):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.idx = 0

    def __len__(self) -> int:
        return len(self.memory)

    def is_ready(self) -> bool:
        return len(self) >= self.batch_size

    def push(self, transition: Transition) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self) -> Batch:
        choices = random.sample(self.memory, self.batch_size)
        x, u, r, y, d = tuple(zip(*choices))  # y is x'

        x, y = torch.stack(x), torch.stack(y)
        u = torch.tensor(u).float()
        r = torch.tensor(r).float().unsqueeze(1)
        d = torch.tensor(d).float().unsqueeze(1)

        return x, u, r, y, d


# Ornstein-Uhlenbeck process

class OrnsteinUhlenbeck:
    '''Ornstein-Uhlenbeck noisy action sampler'''

    def __init__(
        self,
        action_space: gym.spaces.box.Box,
        theta: float = 0.15,
        mu: float = 0.0,
        sigma: float = 0.2
    ):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

        self.low = action_space.low
        self.high = action_space.high

        self.reset()

    def draw(self) -> np.array:
        return np.random.normal(loc=0.0, scale=1.0)

    def reset(self) -> None:
        self.noise = self.draw()

    def action(self, u: np.array) -> np.array:
        self.noise += self.theta * (self.mu - self.noise)
        self.noise += self.sigma * self.draw()

        return np.clip(u + self.noise, self.low, self.high)


# Neural networks

class MLP(nn.Sequential):
    '''Multi-Layer Perceptron'''

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        n_layers: int = 1,
        activation: nn.Module = nn.ReLU
    ):
        layers = [
            nn.Linear(input_size, hidden_size),
            activation(inplace=True)
        ]

        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation(inplace=True))

        layers.append(nn.Linear(hidden_size, output_size))

        super().__init__(*layers)


class Actor(nn.Module):
    '''Actor neural network'''

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        n_layers: int = 1,
        activation: nn.Module = nn.ReLU
    ):
        super().__init__()

        self.mlp = MLP(
            input_size,
            output_size,
            hidden_size,
            n_layers,
            activation
        )
        self.tanh = nn.Tanh()

    def forward(self, x: Batch) -> Batch:
        x = self.mlp(x)

        return self.tanh(x)


class Critic(nn.Module):
    '''Critic neural network'''

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        n_layers: int = 1,
        activation: nn.Module = nn.ReLU
    ):
        super().__init__()

        self.mlp = MLP(
            input_size,
            output_size,
            hidden_size,
            n_layers,
            activation
        )

    def forward(self, x: Batch) -> Batch:
        return self.mlp(x)


# Deep Deterministic Policy Gradient (DDPG) agent

class DDPG:
    '''Deep Deterministic Policy Gradient (DDPG) agent'''

    def __init__(
        self,
        env,
        gamma: float = 0.95,
        tau: float = 1e-2,
        hidden_size: int = 256,
        n_layers: int = 1,
        activation: nn.Module = nn.ReLU,
        capacity: int = 2 ** 16,
        batch_size: int = 32,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        discrete: int = None
    ):
        # Factors

        self.gamma = gamma
        self.tau = tau

        # Networks

        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        self.n_actions = n_actions

        self.actor = Actor(
            n_states,
            n_actions,
            hidden_size,
            n_layers,
            activation
        )
        self.critic = Critic(
            n_states + n_actions,
            n_actions,
            hidden_size,
            n_layers,
            activation
        )

        self.actor_tar = Actor(
            n_states,
            n_actions,
            hidden_size,
            n_layers,
            activation
        )
        self.critic_tar = Critic(
            n_states + n_actions,
            n_actions,
            hidden_size,
            n_layers,
            activation
        )

        self.actor_tar.load_state_dict(self.actor.state_dict())
        self.critic_tar.load_state_dict(self.critic.state_dict())

        # Memory

        self.memory = ReplayBuffer(capacity, batch_size)

        # Training

        self.criterion = nn.MSELoss()
        self.optimizers = {
            'actor': optim.Adam(self.actor.parameters(), lr=lr_actor),
            'critic': optim.Adam(self.critic.parameters(), lr=lr_critic)
        }

        # Discrete action space, if specified

        if discrete is not None:
            linspace = np.linspace(
                start=env.action_space.low,
                stop=env.action_space.high,
                num=discrete
            )
            grid = torch.Tensor(np.meshgrid(*linspace.T))

            self.U = grid.reshape(n_actions, -1).T

            self.knn = NearestNeighbors(n_neighbors=discrete // 2)
            self.knn.fit(self.U)

        self.discrete = discrete

    def wolpertinger(self, x: torch.Tensor) -> torch.Tensor:
        '''Wolpertinger policy [2] used to deal with discrete action space'''

        # Get (continuous) action
        self.actor.eval()

        with torch.no_grad():
            u = self.actor(x)

        # Compute k nearest actions

        idx = self.knn.kneighbors(u, return_distance=False)
        k = self.knn.n_neighbors

        actions = self.U[idx.ravel()]
        actions = actions.view((-1, k, self.n_actions))

        # Choose best action

        stateactions = torch.cat((
            x.repeat_interleave(k, dim=0),
            actions.view((-1, self.n_actions))
        ), dim=1)

        self.critic.train()
        best = self.critic(stateactions).reshape((-1, k)).argmax(axis=1)

        return actions[range(len(actions)), best]

    def action(self, x: np.array) -> np.array:
        '''Get an action used trained network'''

        x = torch.from_numpy(x).float().unsqueeze(0)

        self.actor.eval()

        # Discrete case
        if self.discrete is not None:
            u = self.wolpertinger(x)

        # Continous case
        else:
            with torch.no_grad():
                u = self.actor(x)

        return u.numpy()

    def optimize(self) -> None:
        '''Optimize networks'''

        # Get batches

        x, u, r, x_prime, done = self.memory.sample()

        # Update Q-function by one step of gradient descent

        self.critic.train()
        q = self.critic(torch.cat([x, u], 1))

        if self.discrete is not None:
            u_prime = self.wolpertinger(x_prime)
        else:
            with torch.no_grad():
                self.actor_tar.eval()
                u_prime = self.actor_tar(x_prime)

        self.critic_tar.train()
        q_max = self.critic_tar(torch.cat([x_prime, u_prime], 1))
        target = r + (1 - done) * self.gamma * q_max

        critic_loss = self.criterion(q, target)

        # Update policy by one step of gradien ascent

        policy_loss = -self.critic(torch.cat([x, self.actor(x)], 1)).mean()

        # Optimize

        self.optimizers['actor'].zero_grad()
        policy_loss.backward()
        self.optimizers['actor'].step()

        self.optimizers['critic'].zero_grad()
        critic_loss.backward()
        self.optimizers['critic'].step()

        # Update target networks (Polyak Averaging)

        actor_parameters = zip(
            self.actor_tar.parameters(),
            self.actor.parameters()
        )
        critic_parameters = zip(
            self.critic_tar.parameters(),
            self.critic.parameters()
        )

        for param_tar, param in actor_parameters:
            param_tar.data.copy_(
                param.data * self.tau + param_tar.data * (1.0 - self.tau)
            )

        for param_tar, param in critic_parameters:
            param_tar.data.copy_(
                param.data * self.tau + param_tar.data * (1.0 - self.tau)
            )


########
# Main #
########

def main(
    render: bool = False,
    n_episodes: int = 500,
    discrete: int = None,
    n_layers: int = 1,
    gamma: float = 0.95,
    activation_id: str = 'relu'
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

    max_steps = 1000
    n_evaluate = 50  # number of episodes to evaluate model

    rewards = []

    # Agent

    activations = {
        'relu': nn.ReLU,
        'elu': nn.ELU
    }

    activation = activations.get(activation_id)

    agent = DDPG(
        env,
        gamma=gamma,
        discrete=discrete,
        n_layers=n_layers,
        activation=activation
    )
    noise = OrnsteinUhlenbeck(env.action_space)

    # Training

    for _ in tqdm(range(n_episodes)):
        x = env.reset()
        noise.reset()

        # Simulate the episode until terminal state or max number of steps

        for step in range(max_steps):
            u = agent.action(x)

            if discrete is None:
                u = noise.action(u)

            x_prime, r, done, _ = env.step(u)

            # Save transition

            agent.memory.push((
                torch.tensor(x).float(),
                u[0],
                r,
                torch.tensor(x_prime).float(),
                done
            ))

            # Optimization

            if agent.memory.is_ready():
                agent.optimize()

            x = x_prime

            # If terminal state, stop the episode

            if done:
                break

        # Evaluation

        evals = []

        for _ in range(n_evaluate):
            x = env.reset()
            cr = 0  # cumulative reward

            for step in range(max_steps):
                u = agent.action(x)
                x, r, done, _ = env.step(u)

                if done:
                    break

                cr += (gamma ** step) * r

            evals.append(cr)

        rewards.append(evals)

    # Export results

    rewards = np.array(rewards)

    mean = np.mean(rewards, axis=1)
    std = np.std(rewards, axis=1)

    plt.plot(mean)
    plt.fill_between(
        range(n_episodes),
        mean - std,
        mean + std,
        alpha=0.3
    )

    plt.xlabel('Episode')
    plt.ylabel(r'$J^{\mu}$')

    plt.savefig(f'ddpg_J_{discrete}_{n_layers}_{gamma}.pdf')
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-render', default=False, action='store_true')
    parser.add_argument('-episodes', type=int, default=500)
    parser.add_argument('-discrete', type=int, default=None)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-gamma', type=float, default=0.95)
    parser.add_argument('-activation', type=str, default='relu', choices=['relu', 'elu'])

    args = parser.parse_args()

    main(
        render=args.render,
        n_episodes=args.episodes,
        discrete=args.discrete,
        n_layers=args.layers,
        gamma=args.gamma,
        activation_id=args.activation
    )
