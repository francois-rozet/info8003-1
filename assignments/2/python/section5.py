#!usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from section4 import *


# 5.a Neural Network

class MLP(nn.Sequential):
    """PyTorch Multi-Layer Perceptron"""

    def __init__(
        self,
        input_size: int = 3,
        output_size: int = 1,
        hidden_size: int = 8,
        n_layers: int = 3,
        activation: nn.Module = nn.ReLU
    ):
        layers = [nn.Linear(input_size, hidden_size), activation()]

        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())

        layers.append(nn.Linear(hidden_size, output_size))

        super().__init__(*layers)


# 5.b Parametric Q-learning

def ts_loader(ts: TrainingSet, batch_size: int = 1024) -> data.DataLoader:
    '''Traning set loader'''

    class TSDataset(data.Dataset):
        def __init__(self, ts: TrainingSet):
            stateaction, reward, state_prime = (torch.tensor(x).float() for x in ts)

            self.stateaction, self.reward = stateaction, reward
            self.stateaction_prime = torch.cat([  # all combinations of x' and u'
                state_prime.unsqueeze(1).expand(-1, len(U), -1),
                torch.tensor(U).expand(len(state_prime), -1).unsqueeze(-1)
            ], dim=-1)

        def __len__(self) -> int:
            return len(self.stateaction)

        def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.stateaction[i], self.reward[i], self.stateaction_prime[i]

    dataset = TSDataset(ts)
    loader = data.DataLoader(
        dataset,
        batch_size=None,
        sampler=data.BatchSampler(data.RandomSampler(dataset), batch_size, False),
        pin_memory=torch.cuda.is_available()
    )

    return loader


def dql_epoch(model: nn.Module, goal: nn.Module, loader: data.DataLoader, optimizer: optim.Optimizer):
    '''Double Q-learning epoch'''

    for xu, r, xu_prime in loader:
        xu = xu.to(device)
        q = model(xu).view(-1)

        with torch.no_grad():
            r, xu_prime = r.to(device), xu_prime.to(device)

            max_q = goal(xu_prime).max(dim=1)[0].view(-1)
            target = torch.where(r != 0, r, gamma * max_q)

        loss = F.mse_loss(q, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def pql(model: nn.Module, ts: TrainingSet, epochs: int):
    '''Parametric Q-learning training'''

    loader = ts_loader(ts)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in tqdm.tqdm(range(epochs)):
        dql_epoch(model, model, loader, optimizer)


def dql(model: nn.Module, ts: TrainingSet, epochs: int, passes: int = 5):
    '''Double Q-learning training'''

    loader = ts_loader(ts)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in tqdm.tqdm(range(epochs)):
        goal = model.__class__().to(device)
        goal.load_state_dict(model.state_dict())

        for _ in range(passes):
            dql_epoch(model, goal, loader, optimizer)


## BONUS: Normalized Gradients

def norm_grad_(parameters: Iterable[torch.Tensor], norm: float = 1., norm_type: float = 2.) -> torch.Tensor:
    r"""Normalize gradient of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm (float or int): new norm of the gradients
        norm_type (float or int): type of the used p-norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).

    References:
        https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """

    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.)

    device = parameters[0].grad.device

    total_norm = torch.norm(torch.stack([
        torch.norm(p.grad.detach(), norm_type).to(device)
        for p in parameters
    ]), norm_type)

    coef = norm / (total_norm + 1e-6)
    for p in parameters:
        p.grad.detach().mul_(coef.to(p.grad.device))

    return total_norm


# 5.c Apply

if __name__ == '__main__':
    from plots import plt

    ## N & N'

    N = math.ceil(math.log((eps / (2 * B_r)) * (1. - gamma), gamma))
    N_prime = math.ceil(math.log((eps / B_r), gamma))

    ## Mesh

    p = np.linspace(-1, 1, 200)
    s = np.linspace(-3, 3, 600)

    pp, ss, uu = np.meshgrid(p[:-1], s[:-1], U, indexing='ij')
    stateaction = np.vstack((pp.ravel(), ss.ravel(), uu.ravel())).T
    gridshape = pp.shape

    del pp, ss, uu

    ## Trainings

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # global

    js = {'FQI': [], 'PQL': [], 'DQL': []}
    n = []

    for steps in [25, 50, 100, 150, 200]:
        ts = training_set(exhaustive(steps))
        n.append(len(ts[0]))

        for key, routine in {'FQI': fqi, 'PQL': pql, 'DQL': dql}.items():
            if key == 'FQI':
                model = KMLP()
                fqi(model, ts, N)

                ### Compute Q^_N

                qq = model.predict(stateaction)
            else:
                model = MLP().to(device)

                if key == 'PQL':
                    pql(model, ts, N * 5)
                else:  # key == 'DQL'
                    dql(model, ts, N, 5)

                model.cpu().eval()

                ### Compute Q^

                with torch.no_grad():
                    qq = model(torch.tensor(stateaction).float()).numpy()

            ### Compute mû

            qq = qq.reshape(gridshape)
            mu_hat = 2 * qq.argmax(axis=-1) - 1

            ### Compute J^mû_N'

            trajectories = samples(policify(mu_hat), N_prime)
            j_hat = expected_return(trajectories, N_prime)

            js[key].append(j_hat)

    for key, val in js.items():
        if val:
            plt.plot(n, val, label=key)

    plt.xlabel(r'$n$')
    plt.ylabel(r'$J^\hat{\mu}_N$')
    plt.legend()
    plt.savefig('5_comparison.pdf')
    plt.close()
