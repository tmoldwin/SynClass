"""
Trim synapse volumes to only z-slices that have mask content (pre + post sum > 0).
Slices with no mask are dropped; volume and both masks are resaved in place.
Saves a lot of space (e.g. 64 slices -> 1–10 when masks are sparse).
If masks are all zeros (placeholder), keeps only the center slice per synapse.
"""
import argparse
import os
import numpy as np
from tqdm import tqdm


# SynClass convention: data_3d is (H, W, Z), Z is axis 2
STACK_AXIS = 2


def trim_synapse(data_dir, syn_id, dry_run=False):
    """Trim one synapse to slices with mask content. Returns (slices_before, slices_after, bytes_saved)."""
    base = os.path.join(data_dir, f"{syn_id}_syn")
    syn_path = base + ".npy"
    pre_path = os.path.join(data_dir, f"{syn_id}_pre_syn_n_mask.npy")
    post_path = os.path.join(data_dir, f"{syn_id}_post_syn_n_mask.npy")
    if not os.path.isfile(syn_path):
        return None

    try:
        vol = np.load(syn_path)
    except (ValueError, OSError):
        return None
    if vol.size == 0:
        return None
    try:
        pre = np.load(pre_path)
        post = np.load(post_path)
    except (FileNotFoundError, OSError, ValueError):
        pre = np.zeros_like(vol)
        post = np.zeros_like(vol)

    if vol.shape != pre.shape or vol.shape != post.shape:
        return None

    n_slices = vol.shape[STACK_AXIS]
    # Which slices have any mask?
    mask_sums = [
        float(pre.take(k, axis=STACK_AXIS).sum() + post.take(k, axis=STACK_AXIS).sum())
        for k in range(n_slices)
    ]
    keep = [k for k in range(n_slices) if mask_sums[k] > 0]
    if not keep:
        # All zeros: keep only center slice to save space
        keep = [n_slices // 2]

    if len(keep) >= n_slices:
        return (n_slices, n_slices, 0)

    # Trim along STACK_AXIS
    vol_trim = np.take(vol, keep, axis=STACK_AXIS).copy()
    pre_trim = np.take(pre, keep, axis=STACK_AXIS).copy()
    post_trim = np.take(post, keep, axis=STACK_AXIS).copy()

    before_bytes = vol.nbytes + pre.nbytes + post.nbytes
    after_bytes = vol_trim.nbytes + pre_trim.nbytes + post_trim.nbytes
    saved = before_bytes - after_bytes

    if not dry_run:
        np.save(syn_path, vol_trim)
        np.save(pre_path, pre_trim)
        np.save(post_path, post_trim)

    return (n_slices, len(keep), saved)


def main():
    parser = argparse.ArgumentParser(
        description="Trim synapse stacks to slices with mask content; resave in place."
    )
    parser.add_argument(
        "--data_dir",
        default="Data/proofread_synapses",
        help="Directory with *_syn.npy and mask .npy files",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report what would be trimmed, do not overwrite.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(args.data_dir)

    files = [f for f in os.listdir(args.data_dir) if f.endswith("_syn.npy")]
    syn_ids = [int(f.replace("_syn.npy", "")) for f in files]
    syn_ids.sort()

    total_before_slices = 0
    total_after_slices = 0
    total_saved = 0
    trimmed_count = 0

    for sid in tqdm(syn_ids, desc="Trim" if not args.dry_run else "Dry run"):
        r = trim_synapse(args.data_dir, sid, dry_run=args.dry_run)
        if r is None:
            continue
        before, after, saved = r
        total_before_slices += before
        total_after_slices += after
        total_saved += saved
        if before > after:
            trimmed_count += 1

    print(f"Synapses processed: {len(syn_ids)}")
    print(f"Synapses trimmed:  {trimmed_count}")
    print(f"Slices:            {total_before_slices} -> {total_after_slices}")
    print(f"Space saved:       {total_saved / (1024**3):.2f} GB")
    if args.dry_run:
        print("(Dry run: no files written)")


if __name__ == "__main__":
    main()
