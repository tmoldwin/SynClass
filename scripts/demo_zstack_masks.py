"""
Demo: 8 random synapses × 3 columns (first / mid / last z-slice).
Each cell: EM image with transparent pre (yellow) and post (pink) mask overlay.
Falls back to fetching real masks from CAVE/ImageryClient if on-disk masks are all zeros.
"""
import argparse
import os
import random
import numpy as np
import pandas as pd

STACK_AXIS = 2  # SynClass convention: (H, W, Z)


def fetch_masks_from_cave(sid, ctr, bbox_size, pre_root, post_root):
    """Fetch real pre/post segmentation masks from CAVE ImageryClient."""
    from caveclient import CAVEclient
    import imageryclient as ic
    client = CAVEclient("minnie65_public")
    img_client = ic.ImageryClient(client=client)
    _, seg_dict = img_client.image_and_segmentation_cutout(
        ctr, split_segmentations=True, bbox_size=bbox_size,
    )
    pre_mask = np.array(seg_dict.get(int(pre_root), np.zeros(1)))
    post_mask = np.array(seg_dict.get(int(post_root), np.zeros(1)))
    return pre_mask, post_mask


def main():
    parser = argparse.ArgumentParser(description="Z-stack + transparent masks on EM (8 random synapses × first/mid/last).")
    parser.add_argument("--data_dir", default="Data/proofread_synapses")
    parser.add_argument("--manifest", default="Data/synapses_to_download.csv", help="Manifest with pre/post root IDs (optional)")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out", default="figures/demo_zstack_masks.png")
    parser.add_argument("--fetch_masks", action="store_true", help="Fetch real masks from CAVE if on-disk masks are zeros")
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, "synapse_data.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Need {csv_path}")

    df = pd.read_csv(csv_path)
    e_rows = df[df["pre_clf_type"] == "E"].to_dict("records")
    i_rows = df[df["pre_clf_type"] == "I"].to_dict("records")
    random.seed(42)
    random.shuffle(e_rows)
    random.shuffle(i_rows)
    # Interleave E and I, pick n
    picks = []
    n_e = (args.n + 1) // 2
    n_i = args.n - n_e
    picks.extend(e_rows[:n_e])
    picks.extend(i_rows[:n_i])
    random.shuffle(picks)

    # Try loading manifest for root IDs (needed for live mask fetch)
    manifest = None
    if args.fetch_masks and os.path.isfile(args.manifest):
        mdf = pd.read_csv(args.manifest)
        if "pre_pt_root_id" in mdf.columns and "post_pt_root_id" in mdf.columns:
            manifest = {int(r["id_"]): r for _, r in mdf.iterrows()}

    import matplotlib.pyplot as plt

    n_cols = 3
    col_titles = ["first z", "mid z", "last z"]
    plot_data = []

    for row in picks:
        sid = int(row["id_"])
        syn_path = os.path.join(args.data_dir, f"{sid}_syn.npy")
        pre_path = os.path.join(args.data_dir, f"{sid}_pre_syn_n_mask.npy")
        post_path = os.path.join(args.data_dir, f"{sid}_post_syn_n_mask.npy")
        if not os.path.isfile(syn_path):
            continue
        try:
            data = np.load(syn_path)
        except (ValueError, OSError):
            continue
        if data.size == 0:
            continue
        try:
            pre_mask = np.load(pre_path)
            post_mask = np.load(post_path)
        except (FileNotFoundError, ValueError, OSError):
            pre_mask = np.zeros_like(data)
            post_mask = np.zeros_like(data)

        masks_empty = not pre_mask.any() and not post_mask.any()
        if masks_empty and args.fetch_masks and manifest and sid in manifest:
            mrow = manifest[sid]
            ctr = [int(mrow["ctr_x"]), int(mrow["ctr_y"]), int(mrow["ctr_z"])]
            bbox = (data.shape[0], data.shape[1], data.shape[STACK_AXIS]) if STACK_AXIS == 2 else data.shape
            try:
                pre_mask, post_mask = fetch_masks_from_cave(
                    sid, ctr, bbox,
                    int(mrow["pre_pt_root_id"]), int(mrow["post_pt_root_id"]),
                )
                if pre_mask.ndim < 3:
                    pre_mask = np.zeros_like(data)
                    post_mask = np.zeros_like(data)
            except Exception as e:
                print(f"Could not fetch masks for {sid}: {e}")

        label = row["pre_clf_type"]
        nz = data.shape[STACK_AXIS]
        z_indices = [0, nz // 2, max(0, nz - 1)]
        plot_data.append((data, pre_mask, post_mask, f"{sid} ({label})", z_indices))

    if not plot_data:
        print("No valid synapses found.")
        return

    n_rows = len(plot_data)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("MICrONS synapse crops: EM + masks (Yellow=Pre, Pink=Post)", fontsize=14)

    for r, (data, pre_mask, post_mask, title, z_indices) in enumerate(plot_data):
        for c in range(n_cols):
            z_idx = z_indices[c]
            ax = axes[r, c]
            em = data[:, :, z_idx] if STACK_AXIS == 2 else np.take(data, z_idx, axis=STACK_AXIS)
            em_norm = (em - em.min()) / (em.max() - em.min() + 1e-8)
            ax.imshow(em_norm, cmap="gray")
            h, w = em_norm.shape

            if pre_mask.shape == data.shape:
                pre_sl = pre_mask[:, :, z_idx] if STACK_AXIS == 2 else np.take(pre_mask, z_idx, axis=STACK_AXIS)
                post_sl = post_mask[:, :, z_idx] if STACK_AXIS == 2 else np.take(post_mask, z_idx, axis=STACK_AXIS)
            else:
                pre_sl = np.zeros((h, w))
                post_sl = np.zeros((h, w))

            pre_rgba = np.zeros((h, w, 4))
            post_rgba = np.zeros((h, w, 4))
            pre_rgba[pre_sl.astype(bool)] = [1.0, 1.0, 0.0, 0.6]
            post_rgba[post_sl.astype(bool)] = [1.0, 0.4, 0.7, 0.6]
            ax.imshow(pre_rgba)
            ax.imshow(post_rgba)

            if not pre_sl.any() and not post_sl.any():
                ax.text(0.5, 0.02, "no mask data", transform=ax.transAxes, fontsize=7,
                        ha="center", color="yellow",
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
            ax.axis("off")
        axes[r, 0].set_ylabel(title, fontsize=10, rotation=0, labelpad=60, va="center")

    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
