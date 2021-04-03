#!/usr/bin/env python3

"""
Implementation of Deep Deterministic Policy Gradient (DDPG) algorithm for the
Double Inverted Pendulum control problem.

Source:
    [1] Continuous control with deep reinforcement learning, Lillicrap et al.,
        https://arxiv.org/abs/1509.02971
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

        n_actions = action_space.shape[0]

        self.mean = np.zeros((n_actions))
        self.cov = np.eye((n_actions))

        self.reset()

    def draw(self) -> np.array:
        return np.random.multivariate_normal(self.mean, self.cov)

    def reset(self) -> None:
        self.noise = self.draw()

    def action(self, u: np.array) -> np.array:
        self.noise += self.theta * (self.mu - self.noise)
        self.noise += self.sigma * self.draw()

        return np.clip(u + self.noise, self.low, self.high)


# Neural networks

class Dense(nn.Sequential):
    '''Generic dense layer'''

    def __init__(self, input_size: int, output_size: int):
        super().__init__(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size),
            nn.ReLU(inplace=True)
        )


class MLP(nn.Sequential):
    '''Multi-Layer Perceptron'''

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        n_layers: int = 1
    ):
        layers = [Dense(input_size, hidden_size)]

        for _ in range(n_layers):
            layers.append(Dense(hidden_size, hidden_size))

        layers.append(nn.Linear(hidden_size, output_size))

        super().__init__(*layers)


class Actor(nn.Module):
    '''Actor neural network'''

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        n_layers: int = 1
    ):
        super().__init__()

        self.mlp = MLP(input_size, output_size, hidden_size, n_layers)
        self.tanh = nn.Tanh()

    def forward(self, x: Batch) -> Batch:
        x = self.mlp(x)
        x = self.tanh(x)

        return x


class Critic(nn.Module):
    '''Critic neural network'''

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        n_layers: int = 1
    ):
        super().__init__()

        self.mlp = MLP(input_size, output_size, hidden_size, n_layers)

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
        capacity: int = 2 ** 16,
        batch_size: int = 32,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3
    ):
        # Factors

        self.gamma = gamma
        self.tau = tau

        # Networks

        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        self.actor = Actor(n_states, n_actions, hidden_size)
        self.critic = Critic(n_states + n_actions, n_actions, hidden_size)

        self.actor_tar = Actor(n_states, n_actions, hidden_size)
        self.critic_tar = Critic(n_states + n_actions, n_actions, hidden_size)

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

    def action(self, x: np.array) -> np.array:
        x = torch.from_numpy(x).float().unsqueeze(0)

        self.actor.eval()

        with torch.no_grad():
            u = self.actor.forward(x)
            u = u.numpy()

            return u

    def optimize(self):
        x, u, r, x_prime, done = self.memory.sample()

        # Update Q-function by one step of gradient descent

        self.critic.train()
        q = self.critic.forward(torch.cat([x, u], 1))

        with torch.no_grad():
            self.actor_tar.eval()
            u_prime = self.actor_tar.forward(x_prime)

        self.critic_tar.train()
        q_max = self.critic_tar.forward(torch.cat([x_prime, u_prime], 1))
        target = r + (1 - done) * self.gamma * q_max

        critic_loss = self.criterion(q, target)

        # Update policy by one step of gradien ascent

        policy_loss = -self.critic.forward(
            torch.cat([x, self.actor.forward(x)], 1)
        ).mean()

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
    n_episodes: int = 400
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

    rewards = []
    durations = []

    # Agent

    agent = DDPG(env, gamma=gamma)
    noise = OrnsteinUhlenbeck(env.action_space)

    # Training

    for _ in tqdm(range(n_episodes)):
        x = env.reset()
        noise.reset()

        ecr = 0

        # Simulate the episode until terminal state or max number of steps

        for step in range(max_steps):
            u = agent.action(x)
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

            # Expected cumulative reward

            ecr += (1 - done) * (gamma ** step) * r

            # If terminal state, stop the episode

            if done:
                break

        rewards.append(ecr)
        durations.append(step)

    # Export results

    data = {
        'rewards': {
            'value': rewards,
            'label': 'Expected cumulative reward',
            'file': 'rewards.pdf'
        },
        'durations': {
            'value': durations,
            'label': 'Duration (number of steps)',
            'file': 'durations.pdf'
        }
    }

    for value in data.values():
        plt.plot(value['value'])

        plt.xlabel('Episode')
        plt.ylabel(value['label'])

        plt.savefig(value['file'])
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-render', default=False, action='store_true')
    parser.add_argument('-episodes', type=int, default=400)

    args = parser.parse_args()

    main(
        render=args.render,
        n_episodes=args.episodes,
    )
