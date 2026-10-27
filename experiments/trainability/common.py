from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn import datasets, preprocessing


def iris_binary(samples_per_class=49):
    x = preprocessing.normalize(datasets.load_iris().data)
    A = x[:samples_per_class]
    B = x[50:50 + samples_per_class]
    return np.vstack((A, B)), np.concatenate(
        (np.zeros(len(A), dtype=int), np.ones(len(B), dtype=int))
    )


def historical_qnn_init(seed, d):
    rng = np.random.RandomState(seed)
    rng.rand(d)
    return 2 * np.pi * rng.rand(d)


def qnn_loss(model, theta, x, y):
    output = model.forward(theta, x)
    pa = np.clip(output[:, 0], 1e-12, 1 - 1e-12)
    losses = np.where(y == 0, -np.log(pa), -np.log1p(-pa))
    return float(np.mean(losses))


def qnn_loss_and_grad(model, theta, x, y):
    output, basis_grads = model.forward_and_gradient(theta, x)

    pa = np.clip(output[:, 0], 1e-12, 1 - 1e-12)
    even_indices = next(iter(model.post_processing.values()))
    dpa = basis_grads[:, :, even_indices].sum(axis=2)

    losses = np.where(y == 0, -np.log(pa), -np.log1p(-pa))
    coeff = np.where(y == 0, -1.0 / pa, 1.0 / (1.0 - pa))
    grad = np.mean(coeff[:, None] * dpa, axis=0)

    return float(np.mean(losses)), np.asarray(grad, dtype=float)


def adam_exact(
    loss_grad,
    initial_point,
    maxiter=100,
    lr=0.1,
    beta_1=0.9,
    beta_2=0.99,
    tol=1e-6,
    noise_factor=1e-8,
):
    x = np.asarray(initial_point, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    losses = []

    loss, grad = loss_grad(x)

    for t in range(1, maxiter + 1):
        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * grad * grad

        lr_eff = lr * np.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
        x_new = x - lr_eff * m / (np.sqrt(v) + noise_factor)

        new_loss, new_grad = loss_grad(x_new)
        losses.append(new_loss)

        if np.linalg.norm(x - x_new) < tol:
            x = x_new
            loss = new_loss
            break

        x = x_new
        loss = new_loss
        grad = new_grad

    return x, float(loss), np.asarray(losses, dtype=float)
