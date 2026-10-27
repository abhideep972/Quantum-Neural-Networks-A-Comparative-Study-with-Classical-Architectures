from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[2]


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 1, bias=False),
            nn.LeakyReLU(),
            nn.Linear(1, 1, bias=False),
            nn.LeakyReLU(),
            nn.Linear(1, 1, bias=False),
            nn.LeakyReLU(),
            nn.Linear(1, 2, bias=False),
        ).double()

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    iris = load_iris()
    X = normalize(iris.data[:100])
    y = iris.target[:100]
    X, y = shuffle(X, y, random_state=0)
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.long)

    out = ROOT / 'results' / 'trainability' / 'classical'
    out.mkdir(parents=True, exist_ok=True)

    for run in range(args.runs):
        torch.manual_seed(run)
        model = ANN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()
        losses = []

        for _ in range(args.epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()

        np.save(out / f'classical_loss_{run}.npy', np.asarray(losses))
        print(run, losses[-1])


if __name__ == '__main__':
    main()
