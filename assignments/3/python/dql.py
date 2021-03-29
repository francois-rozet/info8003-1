#!/usr/bin/env python

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import islice
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from display import *
from domain import *


# Types

VisualState = torch.Tensor  # (3, 100, 100)
VisualTransition = Tuple[VisualState, Action, Reward, VisualState]

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# Functions

def state2visual(x: State, factor: int = 4) -> VisualState:
    '''Transforms a state in an image'''

    img = surf2img(draw(*x))
    img = to_tensor(img)
    img = F.avg_pool2d(img, factor)

    return img


## Epsilon-greedy policy

def epsilon(
    step: int,
    start: float = 0.95,
    end: float = 0.05,
    decay: int = 25
) -> float:
    '''Compute the value of epsilon used in epsilon-greedy policy'''

    return end + (start - end) * np.exp(-step / decay)


def greedy(
    eps: float,
    model: nn.Module,
    state: VisualState
) -> Action:
    '''Select an action according to an epsilon-greedy policy'''

    if random.random() > eps:
        with torch.no_grad():
            return model(state.unsqueeze(0)).argmax().item()
    else:
        return random.randrange(2)


## Expected return

def samples(mu: Policy, N: int, n: int = 50, seed: int = 0) -> List[Trajectory]:
    '''Monte Carlo samples'''

    random.seed(seed)

    trajectories = []

    for _ in range(n):
        h = list(islice(simulate(mu), N))  # truncated
        trajectories.append(h)

    return trajectories


def policify(mu: np.array) -> Policy:
    def f(x: State) -> Action:
        p, s = x
        p, s = (p + 1) / 2, (s + 3) / 6
        i, j = int(p * mu.shape[0]), int(s * mu.shape[1])
        i, j = min(i, mu.shape[0] - 1), min(j, mu.shape[1] - 1)

        return mu[i, j]

    return f


def nth(h: Trajectory, N: int) -> Tuple[int, Transition]:
    '''N-th transition or last'''

    if type(h) is list:
        t = min(len(h), N) - 1
        l = h[t]
    else:
        for t, l in zip(range(N), h):
            pass

    return t, l


def cumulative_reward(h: Trajectory, N: int) -> Transition:
    '''Cumulative reward after N steps'''

    t, (_, _, r, _) = nth(h, N)

    return (gamma ** t) * r  # taking advantage of null rewards


def expected_return(trajectories: List[Trajectory], N: int) -> Reward:
    '''Expected return by Monte Carlo'''

    total = sum(cumulative_reward(h, N) for h in trajectories)

    return total / len(trajectories)


# Classes

## Replay buffer

class ReplayBuffer:
    '''Cyclic buffer of bounded capacity'''

    def __init__(self, capacity: int = 2 ** 12, batch_size: int = 128):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.idx = 0

    def __len__(self) -> int:
        return len(self.memory)

    def is_ready(self) -> bool:
        return len(self) >= self.batch_size

    def push(self, transition: VisualTransition) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self) -> Batch:
        choices = random.sample(self.memory, self.batch_size)
        x, u, r, y = tuple(zip(*choices))  # y is x'

        x, y = torch.stack(x), torch.stack(y)
        u, r = torch.tensor(u), torch.tensor(r)

        return x, u, r, y


## Neural network

class Conv(nn.Sequential):
    '''Generic 2D convolution layer'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 2
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DoubleConv(nn.Sequential):
    '''Generic 2D double convolution layer'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 2
    ):
        super().__init__(
            Conv(in_channels, out_channels, kernel_size, 1, padding),
            Conv(out_channels, out_channels, kernel_size, stride, padding),
        )


class MLP(nn.Sequential):
    '''Multi-Layer Perceptron'''

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 8,
        n_layers: int = 3
    ):
        layers = [
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True)
        ]

        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_size, output_size))

        super().__init__(*layers)


class DQN(nn.Module):
    '''Deep Q-Network'''

    def __init__(self, output_size: int = 2, **kwargs):
        super().__init__()

        self.conv1 = DoubleConv(3, 16)
        self.conv2 = DoubleConv(16, 32)
        self.conv3 = DoubleConv(32, 64)
        self.conv4 = DoubleConv(64, 128)

        self.last = MLP(128 * 7 ** 2, output_size, **kwargs)

    def forward(self, x: Batch) -> Batch:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return self.last(x)


# Main

def optimize():
    # Batch

    x, u, r, y = buff.sample()
    x, u, r, y = (
        x.to(device),
        u.to(device),
        r.to(device),
        y.to(device)
    )

    # Compute max_u' Q(x', u')

    with torch.no_grad():
        targetnet.eval()
        q_max = targetnet(y).max(1)[0]
        target = torch.where(r != 0, r, gamma * q_max)

    # Compute Q(x, u)

    model.train()
    q = model(x).gather(1, u.view(-1, 1)).squeeze(1)

    # Optimize

    loss = criterion(q, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-double', default=False, action='store_true')
    args = parser.parse_args()

    # Device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup

    buff = ReplayBuffer()
    model = DQN().to(device)
    targetnet = DQN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Online Deep Q-learning

    with tqdm(total=100 * 25) as tq:
        for epoch in range(100):  # epochs
            if args.double:
                targetnet.load_state_dict(model.state_dict())
            else:
                targetnet = model

            for _ in range(25):  # trajectories
                r, x = 0, initial()
                state = state2visual(x)

                while not terminal(x):
                    ## Take action

                    u = greedy(epsilon(epoch), model, state.to(device))
                    _, _, r, x_prime = onestep(x, u)
                    state_prime = state2visual(x_prime)

                    ### Store in buffer

                    buff.push((state, u, r, state_prime))

                    ### Update state

                    x, state = x_prime, state_prime

                    ## Perform optimization step

                    if buff.is_ready():
                        optimize()

                tq.update(1)

    # Evaluation

    model.eval()

    ## State space

    resolution = 0.01

    P = np.arange(-1., 1., resolution) + resolution / 2
    S = np.arange(-3., 3., resolution) + resolution / 2

    ## Q

    with torch.no_grad():
        Q = []
        buff = []

        for p in P:
            for s in S:
                buff.append(state2visual((p, s)))

                if len(buff) == 512:
                    x = torch.stack(buff).to(device)
                    q = model(x).cpu()
                    Q.append(q)

                    buff = []

        if buff:
            x = torch.stack(buff).to(device)
            q = model(x).cpu()
            Q.append(q)

        Q = torch.cat(Q)
        Q = Q.view(len(P), len(S), len(U)).numpy()

    ## mû

    mu_hat = Q.argmax(axis=-1)

    ## J^mû_N'

    N_prime = math.ceil(math.log((eps / B_r), gamma))

    trajectories = samples(policify(mu_hat), N_prime)
    j_hat = expected_return(trajectories, N_prime)

    print('J^mû_N =', j_hat)
    print()

    ## Save

    name = 'dqn' if args.double else 'dql'

    np.savetxt(f'{name}_q0.txt', Q[..., 0], fmt='%.3e')
    np.savetxt(f'{name}_q1.txt', Q[..., 1], fmt='%.3e')
