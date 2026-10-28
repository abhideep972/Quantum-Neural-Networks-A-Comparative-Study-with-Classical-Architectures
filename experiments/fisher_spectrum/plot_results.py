from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy import interpolate

ROOT = Path.cwd()
DATA = ROOT / "results" / "effective_dimension"
ARCHIVE = ROOT / "results" / "archive"
OUT = ROOT / "figures"
ANALYSIS = ROOT / "results" / "analysis"

ARCHIVE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
ANALYSIS.mkdir(parents=True, exist_ok=True)

paths = {
    "Classical NN": DATA / "fhat4_[4 4 4 2]_fisher.npy",
    "Easy quantum": DATA / "4qubits_9layer_f_hats_pauli.npy",
    "Quantum NN": DATA / "4qubits_9layer_f_hats_dep2.npy",
}

missing = [str(p) for p in paths.values() if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required dataset(s):\n" + "\n".join(missing))

data = {name: np.load(path) for name, path in paths.items()}

# ---------------------------------------------------------------------
# Validate the three canonical full datasets.
# ---------------------------------------------------------------------
summary_lines = []
eig_all = {}

for name, x in data.items():
    if x.shape != (100, 40, 40):
        raise ValueError(f"{name}: expected (100, 40, 40), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: contains non-finite values")

    asym = float(np.max(np.abs(x - x.transpose(0, 2, 1))))
    eig = np.linalg.eigvalsh(x)
    min_eig = float(eig.min())
    max_eig = float(eig.max())
    mean_trace = float(np.trace(x, axis1=1, axis2=2).mean())

    if asym > 1e-10:
        raise ValueError(f"{name}: asymmetry too large: {asym}")
    if min_eig < -1e-10:
        raise ValueError(f"{name}: significantly negative eigenvalue: {min_eig}")
    if not np.isclose(mean_trace, 40.0, rtol=0, atol=1e-8):
        raise ValueError(f"{name}: mean trace is not 40: {mean_trace}")

    eig_all[name] = eig

    summary_lines.append(
        f"{name:14s} shape={x.shape} finite=True "
        f"max_asym={asym:.3e} min_eig={min_eig:.6e} "
        f"max_eig={max_eig:.6e} mean_trace={mean_trace:.12f}"
    )

print("\nVALIDATION")
print("-" * 90)
for line in summary_lines:
    print(line)

(ANALYSIS / "validation_summary.txt").write_text(
    "\n".join(summary_lines) + "\n", encoding="utf-8"
)

# Archive the completed classical dataset.
classical_archive = ARCHIVE / "cnn_full_100x100.npy"
shutil.copy2(paths["Classical NN"], classical_archive)
print(f"\nArchived classical dataset -> {classical_archive}")

# Save raw sorted eigenvalues for later inspection.
for name, eig in eig_all.items():
    safe = name.lower().replace(" ", "_")
    np.save(ANALYSIS / f"{safe}_eigenvalues.npy", eig)

# ---------------------------------------------------------------------
# Effective dimension
# Reproduces the n-grid and normalized formula used in the original
# project plotting script.
# ---------------------------------------------------------------------
n = np.array(
    [1000, 2000, 8000, 10000, 40000, 60000,
     100000, 150000, 200000, 500000, 1000000],
    dtype=float,
)

def normalized_effective_dimension(f_hat, n_values):
    d = f_hat.shape[-1]
    out = []
    for ns in n_values:
        Fhat = f_hat * ns / (2 * np.pi * np.log(ns))
        logdet = np.linalg.slogdet(np.eye(d) + Fhat)[1]
        r = logdet / 2
        eff = 2 * (logsumexp(r) - np.log(len(f_hat))) / np.log(
            ns / (2 * np.pi * np.log(ns))
        )
        out.append(eff / d)
    return np.asarray(out)

ed = {
    name: normalized_effective_dimension(x, n)
    for name, x in data.items()
}

np.savetxt(
    ANALYSIS / "effective_dimension.csv",
    np.column_stack(
        [n, ed["Classical NN"], ed["Easy quantum"], ed["Quantum NN"]]
    ),
    delimiter=",",
    header="n,classical_nn,easy_quantum,quantum_nn",
    comments="",
)

# Colors preserved from the original project plotting scripts.
red = np.array([255, 29, 0]) / 255
blue = np.array([0, 150, 236]) / 255
green = np.array([0, 208, 0]) / 255

plt.figure(figsize=(8, 6))
plt.plot(n, ed["Quantum NN"], label="quantum NN", color=green)
plt.plot(n, ed["Classical NN"], label="classical NN", color=red)
plt.plot(n, ed["Easy quantum"], label="easy quantum", color=blue)
plt.xscale("log")
plt.xlabel("number of data")
plt.ylabel("normalised effective dimension")
plt.ylim(top=1)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "effective_dimension.png", dpi=250)
plt.close()

# ---------------------------------------------------------------------
# Fisher spectra
# The historical project plotting code first sorted each Fisher matrix's
# eigenvalues and then averaged each eigenvalue rank over 100 samples.
# ---------------------------------------------------------------------
avg_eigs = {
    name: np.mean(eig, axis=0)
    for name, eig in eig_all.items()
}

np.savetxt(
    ANALYSIS / "average_sorted_eigenvalues.csv",
    np.column_stack(
        [
            avg_eigs["Classical NN"],
            avg_eigs["Easy quantum"],
            avg_eigs["Quantum NN"],
        ]
    ),
    delimiter=",",
    header="classical_nn,easy_quantum,quantum_nn",
    comments="",
)

def density_curve(values, restrict_lt_one=False):
    v = np.asarray(values)
    if restrict_lt_one:
        v = v[(v > 0) & (v < 1)]
    if len(v) < 4:
        return None, None

    lo, hi = float(v.min()), float(v.max())
    if np.isclose(lo, hi):
        return np.array([lo]), np.array([1.0])

    counts, bins = np.histogram(v, bins=np.linspace(lo, hi, 40))
    mids = (bins[1:] + bins[:-1]) / 2
    y = counts / max(counts.sum(), 1)

    # Match the historical plotting approach, but fall back gracefully.
    try:
        x_new = np.linspace(mids.min(), mids.max(), 200)
        spline = interpolate.make_interp_spline(mids, y)
        y_new = spline(x_new)
        y_new = np.clip(y_new, 0, None)
        return x_new, y_new
    except Exception:
        return mids, y

titles = ["Classical neural network", "Easy quantum model", "Quantum neural network"]
keys = ["Classical NN", "Easy quantum", "Quantum NN"]
colors = [red, blue, green]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, title, key, color in zip(axes, titles, keys, colors):
    xx, yy = density_curve(avg_eigs[key], restrict_lt_one=False)
    ax.plot(xx, yy, color=color)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
axes[0].set_ylabel("density")
axes[1].set_xlabel("value of the eigenvalues")
fig.tight_layout()
fig.savefig(OUT / "fisher_spectrum.png", dpi=250)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, title, key, color in zip(axes, titles, keys, colors):
    xx, yy = density_curve(avg_eigs[key], restrict_lt_one=True)
    if xx is not None:
        ax.plot(xx, yy, color=color)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("density")
axes[1].set_xlabel("value of the eigenvalues")
fig.tight_layout()
fig.savefig(OUT / "fisher_spectrum_zoom_lt1_new.png", dpi=250)
plt.close(fig)

# ---------------------------------------------------------------------
# Cumulative Fisher spectrum, again following the historical logic.
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, title, key, color in zip(axes, titles, keys, colors):
    v = avg_eigs[key]
    lo, hi = float(v.min()), float(v.max())
    counts, bins = np.histogram(v, bins=np.linspace(lo, hi, 40))
    cumulative = np.cumsum(counts) / max(counts.sum(), 1)
    ax.plot(bins[:-1], cumulative, color=color)
    ax.set_title(title)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("cumulative density")
axes[1].set_xlabel("value of the eigenvalues")
fig.tight_layout()
fig.savefig(OUT / "fisher_cumulative.png", dpi=250)
plt.close(fig)

# ---------------------------------------------------------------------
# Presentation-style 2x3 eigenvalue boxplot.
# Each column is a model; top row shows all 40 sorted eigenvalue ranks
# across the 100 sampled Fisher matrices, bottom row zooms to < 1.
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for col, (title, key, color) in enumerate(zip(titles, keys, colors)):
    eig = eig_all[key]  # shape (100, 40), sorted ascending per sample

    axes[0, col].boxplot(
        [eig[:, j] for j in range(eig.shape[1])],
        showfliers=True,
        widths=0.6,
    )
    axes[0, col].set_title(title)
    axes[0, col].set_xlabel("Eigenvalue rank")

    # Sparse rank ticks for readability.
    rank_ticks = sorted(set(
        [1]
        + list(range(5, eig.shape[1] + 1, 5))
        + [eig.shape[1]]
    ))
    axes[0, col].set_xticks(rank_ticks)
    axes[0, col].set_xticklabels(rank_ticks)

    # For the zoomed view, preserve rank-wise samples below 1.
    zoom_groups = []
    positions = []
    for j in range(eig.shape[1]):
        vals = eig[:, j]
        vals = vals[(vals > 0) & (vals < 1)]
        if len(vals):
            zoom_groups.append(vals)
            positions.append(j + 1)

    if zoom_groups:
        axes[1, col].boxplot(
            zoom_groups,
            positions=positions,
            showfliers=True,
            widths=0.6,
        )
    axes[1, col].set_ylim(0, 1)
    axes[1, col].set_xlabel("Eigenvalue rank")

    # Preserve the real eigenvalue ranks, but label only sparse ticks.
    if positions:
        first_rank = min(positions)
        last_rank = max(positions)

        zoom_ticks = sorted(set(
            [first_rank]
            + list(range(5, last_rank + 1, 5))
            + [last_rank]
        ))

        zoom_ticks = [
            t for t in zoom_ticks
            if first_rank <= t <= last_rank
        ]

        axes[1, col].set_xticks(zoom_ticks)
        axes[1, col].set_xticklabels(zoom_ticks)

axes[0, 0].set_ylabel("eigenvalue")
axes[1, 0].set_ylabel("eigenvalue (< 1)")
fig.tight_layout()
fig.savefig(OUT / "fisher_information_boxplots.png", dpi=250)
plt.close(fig)

print("\nEFFECTIVE DIMENSION")
print("-" * 90)
for i, ns in enumerate(n.astype(int)):
    print(
        f"n={ns:7d}  "
        f"classical={ed['Classical NN'][i]:.6f}  "
        f"easy={ed['Easy quantum'][i]:.6f}  "
        f"qnn={ed['Quantum NN'][i]:.6f}"
    )

print("\nCreated:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p)

print("\nAnalysis tables:")
for p in sorted(ANALYSIS.glob("*")):
    print(" ", p)

