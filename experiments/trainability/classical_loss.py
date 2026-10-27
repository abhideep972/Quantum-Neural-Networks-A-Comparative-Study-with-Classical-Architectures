from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize

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


def create_rand_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--start-run", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    iris = load_iris()
    X = normalize(iris.data[:100])
    y = iris.target[:100]
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.long)

    out = ROOT / "results" / "trainability" / "regenerated" / "classical"
    out.mkdir(parents=True, exist_ok=True)

    for run in range(args.start_run, args.start_run + args.runs):
        np.random.seed(run)
        torch.manual_seed(run)

        model = ANN()
        model.apply(create_rand_params)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()
        losses = []

        for _ in range(args.epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()

        np.save(out / f"classical_loss_{run}.npy", np.asarray(losses))
        print(f"run={run:03d} epochs={len(losses):03d} final_loss={losses[-1]:.12f}")


if __name__ == "__main__":
    main()
