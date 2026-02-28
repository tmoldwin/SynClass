"""
Demo: 8 random synapses, full z-stack. Each row = one synapse, columns = evenly spaced z-slices.
"""
import argparse
import os
import random
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Show raw EM z-stacks for 8 random synapses.")
    parser.add_argument("--data_dir", default="Data/proofread_synapses")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max_cols", type=int, default=8, help="Max z-slices to show per synapse")
    parser.add_argument("--out", default="figures/demo_raw_zstack.png")
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, "synapse_data.csv")
    df = pd.read_csv(csv_path)
    e_rows = df[df["pre_clf_type"] == "E"].to_dict("records")
    i_rows = df[df["pre_clf_type"] == "I"].to_dict("records")
    random.seed(123)
    random.shuffle(e_rows)
    random.shuffle(i_rows)
    picks = []
    for a, b in zip(e_rows, i_rows):
        picks.append(a)
        picks.append(b)
    picks = picks[:args.n]

    import matplotlib.pyplot as plt

    plot_data = []
    for row in picks:
        sid = int(row["id_"])
        path = os.path.join(args.data_dir, f"{sid}_syn.npy")
        if not os.path.isfile(path):
            continue
        try:
            vol = np.load(path)
        except (ValueError, OSError):
            continue
        if vol.size == 0:
            continue
        label = row["pre_clf_type"]
        nz = vol.shape[2]
        indices = np.linspace(0, nz - 1, min(args.max_cols, nz)).astype(int)
        indices = list(dict.fromkeys(indices))  # dedupe preserving order
        plot_data.append((vol, f"{sid} ({label}) [{vol.shape}]", indices))

    if not plot_data:
        print("No valid synapses found.")
        return

    n_rows = len(plot_data)
    n_cols = max(len(d[2]) for d in plot_data)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle(f"Raw EM z-stacks ({n_rows} synapses, shape = H×W×Z)", fontsize=14)

    for r, (vol, title, indices) in enumerate(plot_data):
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(indices):
                z_idx = indices[c]
                em = vol[:, :, z_idx]
                em_norm = (em - em.min()) / (em.max() - em.min() + 1e-8)
                ax.imshow(em_norm, cmap="gray")
                if r == 0:
                    ax.set_title(f"z={z_idx}", fontsize=9)
            else:
                ax.axis("off")
                continue
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(title, fontsize=7, rotation=0, labelpad=5, va="center", ha="right")

    plt.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.02, hspace=0.3, wspace=0.05)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=120)
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
