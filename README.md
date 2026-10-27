# Quantum Neural Networks: A Comparative Study with Classical Architectures

Modernized research code for a comparative study of classical neural networks and quantum neural networks (QNNs), developed from work carried out during a quantum-machine-learning research project at IISER Bhopal.

The project studies how model geometry and quantum feature-map choice affect model complexity and trainability through:

- parameterized quantum circuits and variational models;
- Fisher information matrices and Fisher-information spectra;
- effective dimension;
- classical vs. easy-QNN vs. more expressive QNN comparisons;
- trainability and barren-plateau behaviour.

## What is in this repository

```text
qnn_repro/                    modern implementation
experiments/
  effective_dimension/        effective-dimension experiment drivers
  fisher_spectrum/            Fisher-spectrum experiment drivers
  trainability/               classical/QNN training experiment drivers
figures/                      representative reproduced figures
tests/                        smoke tests
.github/workflows/            CI test workflow
requirements.txt
ATTRIBUTION.md
```

The original Abbas *et al.* source tree is intentionally **not vendored here**. It is linked in `ATTRIBUTION.md` instead. This keeps the repository focused on the modern implementation and the analyses used in this project.

## Modern implementation

`qnn_repro/` contains:

- `quantum_nn.py` — statevector QNN evaluation, parameter-shift gradients and Fisher matrices;
- `classical_nn.py` — classical feed-forward baseline;
- `effective_dimension.py` — normalized Fisher matrices and effective-dimension calculation;
- `optimizer.py` — small self-contained Adam optimizer replacing the removed `qiskit.aqua` optimizer;
- `model_base.py` — common model interface.

The QNN implementation uses current Qiskit circuit composition and statevector APIs and avoids the legacy multiprocessing design so it can run on Windows as well as Linux.

## Experiments

### Effective dimension

```bash
python experiments/effective_dimension/ed_cnn.py --num-inputs 10 --num-thetas 10
python experiments/effective_dimension/ed_easy_qnn.py --num-inputs 10 --num-thetas 10 --blocks 2
python experiments/effective_dimension/ed_qnn.py --num-inputs 10 --num-thetas 10 --blocks 2
```

### Fisher-information spectrum

```bash
python experiments/fisher_spectrum/fisher_cnn.py --num-inputs 10 --num-thetas 10
python experiments/fisher_spectrum/fisher_easy_qnn.py --num-inputs 10 --num-thetas 10 --blocks 2
python experiments/fisher_spectrum/fisher_qnn.py --num-inputs 10 --num-thetas 10 --blocks 2
```

### Trainability

Small smoke runs can be used before full experiments:

```bash
python experiments/trainability/classical_loss.py --runs 2 --epochs 10
python experiments/trainability/easy_qnn.py --runs 1 --maxiter 5 --samples-per-class 5
python experiments/trainability/hard_qnn.py --runs 1 --maxiter 5 --samples-per-class 5
```

Generated numerical results are written under `results/`, which is ignored by Git.

## Representative figures

### Effective dimension

![Effective dimension](figures/effective_dimension.png)

### Fisher-information spectrum

![Fisher spectrum](figures/fisher_spectrum.png)

### Cumulative Fisher spectrum

![Cumulative Fisher spectrum](figures/fisher_cumulative.png)

### Trainability comparison

![Trainability](figures/trainability_loss.png)

These figures document the analyses reproduced during the project. They are included as reference outputs; regenerating them exactly requires the full experiment settings and stochastic runs used for the corresponding analysis.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
# activate the environment
pip install -r requirements.txt
```

## Tests

```bash
pytest -q
```

The smoke tests check QNN forward probabilities, parameter-shift gradients, Fisher matrices, and a small effective-dimension calculation. The same tests are run automatically by GitHub Actions.

## Research context

**Institution:** IISER Bhopal  
**Advisor:** Dr. Kuntal Roy, EECS  
**Project:** Quantum Neural Networks / comparative study with classical architectures

## Reference

A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, “The power of quantum neural networks,” *Nature Computational Science* 1, 403–409 (2021).
