import itertools
import os

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import StatevectorSimulator
except ImportError:
    StatevectorSimulator = None

from .model_base import Model


class QuantumNeuralNetwork(Model):
    """Statevector QNN for modern Qiskit, with batched Qiskit Aer acceleration.

    The public API is the same as the previously validated implementation.
    When qiskit-aer is installed, all parameter/input samples in a batch are
    submitted as a parameter sweep of one transpiled circuit. Aer then runs
    independent experiments across CPU cores.

    Environment variables
    ---------------------
    QNN_AER_THREADS
        Maximum CPU threads Aer may use. 0 (default) means all available cores.
    QNN_AER_DEVICE
        "CPU" (default) or "GPU" if using a GPU-enabled Aer build.
    QNN_AER_BATCH
        Optional maximum number of parameterizations per Aer job. 0 (default)
        submits the entire batch in one job. Set e.g. 5000 only if memory use
        becomes a problem.
    """

    def __init__(self, feature_map, var_form, post_processing=None, use_aer=True):
        super().__init__()

        if feature_map.num_qubits != var_form.num_qubits:
            raise ValueError(
                "feature map and variational form must use the same qubit count"
            )

        self.feature_map = feature_map
        self.var_form = var_form
        self.feature_parameters = list(feature_map.parameters)
        self.var_parameters = list(var_form.parameters)
        self.d = len(self.var_parameters)
        self.inputsize = len(self.feature_parameters)

        self.circuit = QuantumCircuit(feature_map.num_qubits)
        self.circuit.compose(feature_map, inplace=True)
        self.circuit.compose(var_form, inplace=True)

        self.post_processing = post_processing or self._parity()
        self.outputsize = len(self.post_processing)

        self.using_aer = bool(use_aer and StatevectorSimulator is not None)
        self._aer_backend = None
        self._aer_circuit = None
        self._aer_batch = int(os.environ.get("QNN_AER_BATCH", "0"))

        if self.using_aer:
            threads = int(os.environ.get("QNN_AER_THREADS", "0"))
            device = os.environ.get("QNN_AER_DEVICE", "CPU").upper()

            self._aer_backend = StatevectorSimulator(
                device=device,
                max_parallel_threads=threads,
                max_parallel_experiments=0,
            )

            # Transpile the parameterized circuit once. The Parameter objects
            # remain bindable, so subsequent calls only sweep numerical values.
            self._aer_circuit = transpile(
                self.circuit,
                self._aer_backend,
                optimization_level=1,
            )

    def _get_params_dict(self, params, x):
        if len(x) != len(self.feature_parameters):
            raise ValueError("input dimension does not match feature map")
        if len(params) != len(self.var_parameters):
            raise ValueError(
                "parameter dimension does not match variational form"
            )

        return {
            **dict(zip(self.feature_parameters, x)),
            **dict(zip(self.var_parameters, params)),
        }

    @staticmethod
    def _as_batches(params, x):
        p = np.asarray(params, dtype=float)
        d = np.asarray(x, dtype=float)

        if p.ndim == 1:
            p = p[None, :]
        if d.ndim == 1:
            d = d[None, :]

        if len(p) == 1 and len(d) > 1:
            p = np.repeat(p, len(d), axis=0)
        if len(d) == 1 and len(p) > 1:
            d = np.repeat(d, len(p), axis=0)

        if len(p) != len(d):
            raise ValueError(
                "params and x must have equal batch sizes "
                "or one batch size must be 1"
            )

        return p, d

    def _serial_probabilities(self, params, x):
        """Reference implementation using Qiskit's pure Statevector path."""
        probs = np.empty(
            (len(x), 2 ** self.circuit.num_qubits),
            dtype=float,
        )

        for i, (theta, data) in enumerate(zip(params, x)):
            bound = self.circuit.assign_parameters(
                self._get_params_dict(theta, data),
                inplace=False,
            )
            probs[i] = Statevector.from_instruction(bound).probabilities()

        return probs

    def _aer_probability_chunk(self, params, x):
        """Evaluate one batch with one Aer parameter-sweep job."""
        bind = {}

        for j, parameter in enumerate(self.feature_parameters):
            bind[parameter] = x[:, j].tolist()

        for j, parameter in enumerate(self.var_parameters):
            bind[parameter] = params[:, j].tolist()

        result = self._aer_backend.run(
            [self._aer_circuit],
            parameter_binds=[bind],
        ).result()

        probs = np.empty(
            (len(x), 2 ** self.circuit.num_qubits),
            dtype=float,
        )

        for i in range(len(x)):
            state = np.asarray(result.data(i)["statevector"], dtype=complex)
            probs[i] = np.abs(state) ** 2

        return probs

    def _aer_probabilities(self, params, x):
        """Evaluate a batch with Aer, optionally splitting very large jobs."""
        n = len(x)

        if self._aer_batch <= 0 or n <= self._aer_batch:
            return self._aer_probability_chunk(params, x)

        out = np.empty(
            (n, 2 ** self.circuit.num_qubits),
            dtype=float,
        )

        for start in range(0, n, self._aer_batch):
            stop = min(start + self._aer_batch, n)
            out[start:stop] = self._aer_probability_chunk(
                params[start:stop],
                x[start:stop],
            )

        return out

    def _get_probabilities(self, params, x):
        params, x = self._as_batches(params, x)

        if self.using_aer:
            return self._aer_probabilities(params, x)

        return self._serial_probabilities(params, x)

    def forward(self, params, x):
        probs = self._get_probabilities(params, x)
        out = np.zeros((len(probs), self.outputsize), dtype=float)

        for j, indices in enumerate(self.post_processing.values()):
            out[:, j] = probs[:, indices].sum(axis=1)

        return out

    def get_gradient(self, params, x):
        params, x = self._as_batches(params, x)
        n = len(x)

        grads = np.empty(
            (n, self.d, 2 ** self.circuit.num_qubits),
            dtype=float,
        )

        # One Aer job per trainable parameter. Plus/minus parameter-shift
        # evaluations are concatenated into the same parameter sweep, reducing
        # backend submission overhead by ~2x versus separate jobs.
        for j in range(self.d):
            plus = params.copy()
            minus = params.copy()

            plus[:, j] += np.pi / 2
            minus[:, j] -= np.pi / 2

            shifted_params = np.concatenate((plus, minus), axis=0)
            shifted_x = np.concatenate((x, x), axis=0)

            shifted_probs = self._get_probabilities(
                shifted_params,
                shifted_x,
            )

            grads[:, j, :] = 0.5 * (
                shifted_probs[:n] - shifted_probs[n:]
            )

        return grads

    def get_fisher(self, gradients, model_output):
        gradients = np.asarray(gradients, dtype=float)
        output = np.asarray(model_output, dtype=float)

        fishers = np.zeros(
            (len(gradients), self.d, self.d),
            dtype=float,
        )
        eps = np.finfo(float).eps

        for k in range(len(gradients)):
            for label_idx, indices in enumerate(
                self.post_processing.values()
            ):
                p = max(float(output[k, label_idx]), eps)

                # Keep the parameter axis first. Using gradients[k, :, indices]
                # triggers NumPy advanced-index axis reordering.
                dp = gradients[k][:, indices].sum(axis=1)

                score = dp / np.sqrt(p)
                fishers[k] += np.outer(score, score)

        return fishers

    def _parity(self):
        even = []
        odd = []

        for idx, bits in enumerate(
            itertools.product(
                [0, 1],
                repeat=self.circuit.num_qubits,
            )
        ):
            if sum(bits) % 2 == 0:
                even.append(idx)
            else:
                odd.append(idx)

        return {"y1": even, "y2": odd}
