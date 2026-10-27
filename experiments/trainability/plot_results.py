from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def load_runs(directory, pattern, runs, expected_len):
    rows = []
    for i in range(runs):
        path = directory / pattern.format(i=i)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        x = np.asarray(np.load(path), dtype=float)
        if len(x) != expected_len:
            raise ValueError(f"{path}: expected {expected_len}, found {len(x)}")
        if not np.isfinite(x).all():
            raise ValueError(f"{path}: contains non-finite values")
        rows.append(x)
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    base = ROOT / "results" / "trainability" / "regenerated"
    classical = load_runs(
        base / "classical", "classical_loss_{i}.npy", args.runs, args.iterations
    )
    easy = load_runs(
        base / "easy_qnn", "quantum_loss_easy_{i}.npy", args.runs, args.iterations
    )
    hard = load_runs(
        base / "qnn", "quantum_loss_hard_dep2_{i}.npy", args.runs, args.iterations
    )

    red = np.array([255, 29, 0]) / 255
    blue = np.array([0, 150, 236]) / 255
    green = np.array([0, 208, 0]) / 255

    fig, ax = plt.subplots(figsize=(8, 6))
    steps = np.arange(args.iterations)

    for arr, label, color in (
        (classical, "classical neural network", red),
        (easy, "easy quantum model", blue),
        (hard, "quantum neural network", green),
    ):
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(steps, mean, label=label, color=color)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xlabel("number of training iterations")
    ax.set_ylabel("loss value")
    ax.legend()
    fig.tight_layout()

    out = ROOT / "results" / "final_figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "trainability_loss_new.png", dpi=250)
    plt.close(fig)

    print("SUMMARY")
    for name, arr in (
        ("classical", classical),
        ("easy_qnn", easy),
        ("qnn", hard),
    ):
        print(
            f"{name:10s} initial_mean={arr[:,0].mean():.9f} "
            f"final_mean={arr[:,-1].mean():.9f} "
            f"final_std={arr[:,-1].std():.9f}"
        )
    print("saved", out / "trainability_loss_new.png")


if __name__ == "__main__":
    main()
