# Quantum Neural Networks: A Comparative Study with Classical Architectures

Summer 2025 research-training repository documenting my comparative study and reproduction of selected results from **“The power of quantum neural networks”** (Abbas *et al.*, *Nature Computational Science*, 2021) during my internship at **IISER Bhopal**, advised by **Dr. Kuntal Roy (EECS)**.

## Project scope

The project focused on understanding how the geometry of a model's parameter space can be used to study the **capacity and trainability of quantum neural networks (QNNs)**.

I studied and reproduced selected analyses involving:

- parameterized quantum circuits and variational quantum models;
- quantum feature maps and alternative data-encoding strategies;
- the **Fisher information matrix** and its eigenvalue spectrum;
- **effective dimension** as an information-geometric model-complexity measure;
- comparisons between classical feed-forward networks, an "easy" quantum model, and a more expressive QNN;
- the relationship between Fisher-spectrum concentration, flat optimization landscapes, **trainability**, and barren-plateau behaviour.

The internship report and final presentation used these ideas to compare classical and quantum model parameterizations and to understand how the choice of quantum encoding can alter the optimization landscape.

## Important provenance note

This repository is a **reproduction / research-study record**, not a claim of authorship of the underlying effective-dimension software.

The code under [`upstream_code/`](upstream_code/) and the selected reproduction scripts/data under [`reproduction/`](reproduction/) originate from the open-source repository accompanying:

> A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner,  
> **“The power of quantum neural networks,”** *Nature Computational Science* 1, 403–409 (2021).  
> DOI: 10.1038/s43588-021-00084-1

Original implementation: https://github.com/amyami187/effective_dimension  
Archived software release: https://doi.org/10.5281/zenodo.4732830

The original software is distributed under the **Apache License 2.0**. Its license is retained in [`upstream_code/LICENSE`](upstream_code/LICENSE).

My contribution in this repository is the **internship study/reproduction context, documentation, selection of the analyses used during the project, and the accompanying report/presentation**.

## Repository structure

```text
Quantum-Neural-Networks-A-Comparative-Study-with-Classical-Architectures/
├── README.md
├── ATTRIBUTION.md
├── requirements-legacy.txt
├── .gitignore
│
├── docs/
│   ├── Training_Report.pdf
│   └── Final_Presentation.pdf
│
├── upstream_code/
│   ├── LICENSE
│   ├── README.md
│   ├── example_cnn.py
│   ├── example_qnn.py
│   └── effective_dimension/
│
└── reproduction/
    ├── effective_dimension/
    ├── fisher_spectrum/
    └── trainability/
```

## What each reproduction section corresponds to

### 1. Effective dimension

[`reproduction/effective_dimension/`](reproduction/effective_dimension/) contains the scripts and data used to compare effective dimension across:

- a classical feed-forward neural network;
- an easy quantum model;
- a QNN with a more expressive feature map.

The original reproduction setup uses input size 4, output size 2, and 40 trainable parameters for the effective-dimension comparison.

### 2. Fisher information spectrum

[`reproduction/fisher_spectrum/`](reproduction/fisher_spectrum/) contains the selected main-figure scripts/data for studying the eigenvalue distribution of the Fisher information matrix.

The key interpretation investigated during the internship was that a Fisher spectrum concentrated near zero corresponds to many locally flat parameter directions, while a broader spectrum can indicate a more favourable optimization landscape.

### 3. Trainability

[`reproduction/trainability/`](reproduction/trainability/) contains the selected training scripts and plotting code used in the original study to compare the loss evolution of classical and quantum models.

This part of the internship was used to connect:

```text
feature-map choice
        ↓
Fisher-information spectrum
        ↓
optimization-landscape geometry
        ↓
trainability / barren-plateau behaviour
```

## Internship documents

The repository includes the materials I prepared during the IISER Bhopal project:

- [`docs/Training_Report.pdf`](docs/Training_Report.pdf)
- [`docs/Final_Presentation.pdf`](docs/Final_Presentation.pdf)

These document my study of QNN architecture, feature mapping, Fisher information, effective dimension, Fisher spectra, and trainability.

## Legacy environment

The original reproduction code was written for an older Qiskit/PyTorch stack. A historically compatible environment is approximately:

```bash
conda create -n qnn-repro python=3.7
conda activate qnn-repro
pip install qiskit==0.23.0 torch==1.3.1 numpy scipy matplotlib scikit-learn
```

Modern Qiskit APIs have changed substantially, so the original scripts may require adaptation to run under current versions.

## Example code

The original package includes simple classical and quantum model examples:

```bash
cd upstream_code
python example_cnn.py
python example_qnn.py
```

The QNN example constructs a feature map and variational form, evaluates model probabilities, and computes Fisher-information quantities used by the effective-dimension framework.

## Research context

**Institution:** Indian Institute of Science Education and Research (IISER) Bhopal  
**Period:** Summer 2025  
**Advisor:** Dr. Kuntal Roy, EECS Department  
**Student:** Abhideep Jain, B.Tech. Engineering Physics, Delhi Technological University

## References

1. A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, “The power of quantum neural networks,” *Nature Computational Science* 1, 403–409 (2021). https://doi.org/10.1038/s43588-021-00084-1
2. H. Qi *et al.*, “The barren plateaus of quantum neural networks: review, taxonomy and trends,” *Quantum Information Processing* 22, 435 (2023).
3. V. Havlíček *et al.*, “Supervised learning with quantum-enhanced feature spaces,” *Nature* 567, 209–212 (2019).

## License and reuse

The copied upstream implementation is licensed under Apache-2.0; see [`upstream_code/LICENSE`](upstream_code/LICENSE).  
The internship report and presentation are included as project documentation and are not relicensed by this repository.
