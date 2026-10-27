from pathlib import Path
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from qiskit.circuit.library import z_feature_map, real_amplitudes
from qnn_repro import QuantumNeuralNetwork
from common import iris_binary, historical_qnn_init, qnn_loss_and_grad, adam_exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--start-run", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--samples-per-class", type=int, default=49)
    args = parser.parse_args()

    x, y = iris_binary(args.samples_per_class)
    fm = z_feature_map(feature_dimension=4, reps=1)
    vf = real_amplitudes(4, reps=1)
    model = QuantumNeuralNetwork(fm, vf)

    out = ROOT / "results" / "trainability" / "regenerated" / "easy_qnn"
    out.mkdir(parents=True, exist_ok=True)

    for run in range(args.start_run, args.start_run + args.runs):
        init = historical_qnn_init(run, model.d)
        loss_grad = lambda theta: qnn_loss_and_grad(model, theta, x, y)
        theta, value, loss = adam_exact(
            loss_grad, init, maxiter=args.maxiter, lr=0.1
        )
        np.save(out / f"quantum_loss_easy_{run}.npy", loss)
        np.save(out / f"params_{run}.npy", theta)
        print(f"run={run:03d} iterations={len(loss):03d} final_loss={value:.12f}")


if __name__ == "__main__":
    main()
