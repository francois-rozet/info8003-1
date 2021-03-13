#!usr/bin/env python

import numpy as np
import random
import torch
import torch.nn as nn

from typing import Iterable, Tuple

from display import *
from domain import *


# Types

VisualState = torch.Tensor
VisualTransition = Tuple[VisualState, Action, Reward, VisualState]

Batch = Iterable[torch.Tensor]


# Globals

memory_size = 10000

eps_start = 0.9
eps_end = 0.05
eps_decay = 200

num_episodes = 50
batch_size = 128

target_update = 10


# Classes

## Replay buffer

class ReplayBuffer:
    '''Cyclic buffer of bounded size.'''

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memory = []
        self.idx = 0

    def push(self, transition: VisualTransition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int = 128) -> Tuple[Batch]:
        choices = random.sample(self.memory, batch_size)
        x, u, r, x_prime = [], [], [], []

        for choice in choices:
            x.append(choice[0])
            u.append(torch.tensor([choice[1]], dtype=torch.long))
            r.append(torch.tensor([choice[2]]))
            x_prime.append(choice[3])

        mask = torch.tensor(
            tuple(map(lambda x: x is not None, x_prime)),
            device=device,
            dtype=torch.bool
        )

        filtered = [x for x in x_prime if x is not None]

        x, u = torch.stack(x), torch.stack(u)
        r = torch.cat(tuple(r))
        filtered = torch.stack(filtered)

        return x, u, r, mask, filtered

    def __len__(self) -> int:
        return len(self.memory)


## Neural network

class Conv(nn.Sequential):
    '''Generic convolution layer.'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 0
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DQN(nn.Module):
    '''Deep Q-network.'''

    def __init__(self, output_size: int = 2):
        super().__init__()

        self.conv1 = Conv(3, 16)
        self.conv2 = Conv(16, 32)
        self.conv3 = Conv(32, 32)

        self.last = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2592, output_size)
        )

    def forward(self, x: Batch) -> Batch:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.last(x)

        return x


# Functions

## Epsilon-greedy policy

def eps(step: int) -> float:
    '''Compute the value of epsilon used in epsilon-greedy policy.'''

    eps = eps_end + (eps_start - eps_end)
    eps *= np.exp(-1. * step / eps_decay)

    return eps


def greedy(
    eps: float,
    model: nn.Module,
    state: torch.Tensor
) -> Action:
    '''Select an action according to an epsilon-greedy policy.'''

    if np.random.uniform() > eps:
        with torch.no_grad():
            return model(state).max(1)[1].view(1, 1).item()
    else:
        return np.random.randint(2)


## Training

def train_epoch(
    memory: ReplayBuffer,
    model: nn.Module,
    device: torch.device,
    goal: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer
):
    '''Train network for one epoch.'''

    if len(memory) < batch_size:
        return

    # Sample batches
    x, u, r, mask, x_prime = memory.sample(batch_size)

    x, u, r = x.to(device), u.to(device), r.to(device)
    mask, x_prime = mask.to(device), x_prime.to(device)

    # Compute Q(state, action)
    xu = model(x).gather(1, u)

    # Compute V(state_{t+1}) for all next states
    vx = torch.zeros(batch_size, device=device)

    with torch.no_grad():
        vx[mask] = goal(x_prime).max(1)[0]

    # Compute expected Q values
    target = (vx * gamma) + r
    target = target.unsqueeze(1)

    # Optimize
    loss = criterion(xu, target)

    optimizer.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.grad.data.clamp_(0, 1)

    optimizer.step()


# Main

if __name__ == '__main__':
    import torchvision.transforms as transforms

    from itertools import count
    from tqdm import tqdm

    # Transformation

    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(100),
        transforms.ToTensor()
    ])

    ## Replay buffer

    memory = ReplayBuffer(memory_size)

    ## Networks

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, goal = DQN(len(U)).to(device), DQN(len(U)).to(device)

    goal.load_state_dict(model.state_dict())
    goal.eval()

    ## Optimization

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.RMSprop(model.parameters())

    ## Training

    step = 0

    with PyGameDisplay():
        for idx in tqdm(range(num_episodes)):

            ## Reset environment

            p, s = initial()

            ## Visual state

            state = to_tensor(surf2img(draw(p, s)))

            for _ in count():

                ### Select and perform an action

                u = greedy(eps(step), model, state.unsqueeze(0).to(device))
                _, _, r, (p, s) = onestep((p, s), u)

                step += 1

                ### Update next state

                next_state = to_tensor(surf2img(draw(p, s))) if r == 0 else None

                ### Store the transition in memory

                memory.push((state, u, r, next_state))

                ### Move to the next state

                state = next_state

                ### Perform one step of the optimization (on the target network)

                train_epoch(memory, model, device, goal, criterion, optimizer)

                ### If episode is over, break

                if r != 0:
                    break

            ## Update the target network, copying all weights and biases in DQN

            if idx % target_update == 0:
                goal.load_state_dict(model.state_dict())
