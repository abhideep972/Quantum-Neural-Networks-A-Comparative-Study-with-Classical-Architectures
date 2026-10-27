import numpy as np
from qiskit.circuit.library import z_feature_map, real_amplitudes

from qnn_repro import QuantumNeuralNetwork
from experiments.trainability.common import qnn_loss, qnn_loss_and_grad


def test_trainability_parameter_shift_gradient_matches_finite_difference():
    fm = z_feature_map(feature_dimension=4, reps=1)
    vf = real_amplitudes(4, reps=1)
    model = QuantumNeuralNetwork(fm, vf, use_aer=False)

    x = np.array(
        [
            [0.80, 0.50, 0.20, 0.10],
            [0.55, 0.45, 0.55, 0.45],
        ],
        dtype=float,
    )
    y = np.array([0, 1], dtype=int)
    theta = np.linspace(0.2, 1.6, model.d)

    _, analytic = qnn_loss_and_grad(model, theta, x, y)

    eps = 1e-6
    numeric = np.empty_like(theta)
    for j in range(model.d):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += eps
        minus[j] -= eps
        numeric[j] = (
            qnn_loss(model, plus, x, y) - qnn_loss(model, minus, x, y)
        ) / (2 * eps)

    assert np.allclose(analytic, numeric, rtol=2e-5, atol=2e-6)
