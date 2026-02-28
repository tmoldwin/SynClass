"""
Demo: show synapse crops from Data/proofread_synapses (full z-stack per synapse).
We pull a 3D volume per synapse (default 256x256 xy, 64 z), not one image.
Displays middle z-slice and max-projection for a few E/I examples.
"""
import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="View synapse EM crops (full z-stack per synapse).")
    parser.add_argument("--data_dir", default="Data/proofread_synapses", help="Directory with *_syn.npy")
    parser.add_argument("--n", type=int, default=6, help="Number of synapses to show (3 E + 3 I)")
    parser.add_argument("--out", default="figures/demo_synapse_images.png", help="Output figure path")
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, "synapse_data.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Need {csv_path} (run download_synapse_crops.py first).")

    import pandas as pd
    df = pd.read_csv(csv_path)
    e_ids = df[df["pre_clf_type"] == "E"]["id_"].tolist()
    i_ids = df[df["pre_clf_type"] == "I"]["id_"].tolist()
    n_e = min((args.n + 1) // 2, len(e_ids))
    n_i = min(args.n - n_e, len(i_ids))
    ids = list(e_ids[:n_e]) + list(i_ids[:n_i])

    volumes = []
    labels = []
    for sid in ids:
        path = os.path.join(args.data_dir, f"{int(sid)}_syn.npy")
        if not os.path.isfile(path):
            continue
        arr = np.load(path)
        if arr.size == 0:
            continue
        volumes.append(arr)
        labels.append("E" if sid in e_ids else "I")

    if not volumes:
        print("No valid *_syn.npy found.")
        return

    # Report shape (full z-stack)
    shp = volumes[0].shape
    print(f"Per-synapse volume shape: {shp} (Z, Y, X) = full z-stack, not a single image.")
    print(f"Showing {len(volumes)} synapses: {labels}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to save demo figure: pip install matplotlib")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_show = len(volumes)
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
    if n_show == 1:
        axes = axes.reshape(2, 1)
    for j, (vol, label) in enumerate(zip(volumes, labels)):
        # Middle z-slice
        z_mid = vol.shape[0] // 2
        slice_mid = vol[z_mid]
        axes[0, j].imshow(slice_mid, cmap="gray")
        axes[0, j].set_title(f"{label} mid-z")
        axes[0, j].axis("off")
        # Max projection
        max_proj = np.max(vol, axis=0)
        axes[1, j].imshow(max_proj, cmap="gray")
        axes[1, j].set_title(f"{label} max-z")
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("Mid slice")
    axes[1, 0].set_ylabel("Max proj")
    plt.suptitle(f"Synapse crops (full z-stack per synapse, shape {shp})")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
