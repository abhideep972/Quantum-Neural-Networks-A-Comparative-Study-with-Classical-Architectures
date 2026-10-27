import numpy as np
from qiskit.circuit.library import real_amplitudes, z_feature_map
from qiskit.quantum_info import Statevector

from qnn_repro import QuantumNeuralNetwork


def test_aer_batched_matches_reference_statevector():
    q = QuantumNeuralNetwork(
        z_feature_map(2, reps=1),
        real_amplitudes(2, reps=1),
    )

    assert q.using_aer, "qiskit-aer is not installed/active"

    x = np.array([
        [0.10, 0.20],
        [0.25, -0.15],
        [-0.30, 0.40],
    ])
    params = np.array([
        [0.10, 0.20, 0.30, 0.40],
        [0.40, 0.30, 0.20, 0.10],
        [-0.20, 0.15, -0.10, 0.05],
    ])

    aer_probs = q._get_probabilities(params, x)

    reference = []
    for theta, data in zip(params, x):
        bound = q.circuit.assign_parameters(
            q._get_params_dict(theta, data),
            inplace=False,
        )
        reference.append(
            Statevector.from_instruction(bound).probabilities()
        )

    reference = np.asarray(reference)

    assert np.allclose(aer_probs, reference, atol=1e-12, rtol=1e-12)
