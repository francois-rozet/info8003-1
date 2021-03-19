#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def eps(
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

        self.conv1 = Conv(3, 16)
        self.conv2 = Conv(16, 32)
        self.conv3 = Conv(32, 32)

        self.last = MLP(32 * 13 ** 2, output_size, **kwargs)

    def forward(self, x: Batch) -> Batch:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

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
    # Device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Setup

    DOUBLE = False  # Double Q-learning or not

    buff = ReplayBuffer()
    model = DQN().to(device)
    targetnet = DQN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Online Deep Q-learning

    for epoch in tqdm(range(100)):  # epochs
        if DOUBLE:
            targetnet.load_state_dict(model.state_dict())
        else:
            targetnet = model

        for _ in range(5):  # trajectories
            r, x = 0, initial()
            state = state2visual(x)

            while not terminal(x):
                ## Take action

                u = greedy(eps(epoch), model, state.to(device))
                _, _, r, x_prime = onestep(x, u)
                state_prime = state2visual(x_prime)

                ### Store in buffer

                buff.push((state, u, r, state_prime))

                ### Update state

                x, state = x_prime, state_prime

                ## Perform optimization step

                if buff.is_ready():
                    optimize()

    # Evaluation

    ## State space

    resolution = 0.02

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

    ## Save

    name = 'dqn' if DOUBLE else 'dql'

    np.savetxt(f'{name}_left.txt', Q[..., 0], fmt='%.3e')
    np.savetxt(f'{name}_right.txt', Q[..., 1], fmt='%.3e')
