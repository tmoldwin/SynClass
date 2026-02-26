"""
Download EM cutouts for synapses listed in synapses_to_download.csv.
Saves volumes in SynClass format: {id}_syn.npy, {id}_pre_syn_n_mask.npy, {id}_post_syn_n_mask.npy
(masks are zeros; SynClass expects 3-channel input). Writes synapse_data.csv (id_, pre_clf_type).

Requires: pip install cloud-volume pandas numpy tqdm
"""

import argparse
import os

import numpy as np
import pandas as pd
from cloudvolume import CloudVolume
from tqdm import tqdm


EM_CLOUDPATH = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"


def compute_bbox(cx, cy, cz, half_xy, half_z):
    """Return (z0, z1), (y0, y1), (x0, x1) in voxel indices (4,4,40 nm)."""
    cx, cy, cz = int(cx), int(cy), int(cz)
    z0, z1 = max(cz - half_z, 0), cz + half_z
    y0, y1 = max(cy - half_xy, 0), cy + half_xy
    x0, x1 = max(cx - half_xy, 0), cx + half_xy
    return (z0, z1), (y0, y1), (x0, x1)


def main():
    parser = argparse.ArgumentParser(
        description="Download synapse EM crops in SynClass format (id_syn.npy + zero masks)."
    )
    parser.add_argument(
        "--manifest_csv",
        default="Data/synapses_to_download.csv",
        help="CSV from filter_synapses_ei.py: id_, pre_clf_type, ctr_x, ctr_y, ctr_z",
    )
    parser.add_argument(
        "--out_dir",
        default="Data/proofread_synapses",
        help="Output directory: id_syn.npy, id_pre_syn_n_mask.npy, id_post_syn_n_mask.npy, synapse_data.csv",
    )
    parser.add_argument(
        "--cube_xy",
        type=int,
        default=256,
        help="Cube half-size in x,y (voxels).",
    )
    parser.add_argument(
        "--cube_z",
        type=int,
        default=64,
        help="Cube half-size in z (voxels).",
    )
    parser.add_argument(
        "--max_synapses",
        type=int,
        default=None,
        help="Max number to download (default: all in manifest).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.manifest_csv):
        raise FileNotFoundError(
            f"Manifest not found: {args.manifest_csv}\n"
            "Run filter_synapses_ei.py first."
        )

    df = pd.read_csv(args.manifest_csv)
    for c in ["id_", "pre_clf_type", "ctr_x", "ctr_y", "ctr_z"]:
        if c not in df.columns:
            raise ValueError(f"Manifest must have column '{c}'")
    if args.max_synapses is not None:
        df = df.head(args.max_synapses)
    os.makedirs(args.out_dir, exist_ok=True)

    half_xy = args.cube_xy // 2
    half_z = args.cube_z // 2

    print("Opening EM volume...")
    vol = CloudVolume(EM_CLOUDPATH, mip=0, cache=False, progress=True, fill_missing=True)

    done = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        sid = int(row["id_"])
        cx, cy, cz = row["ctr_x"], row["ctr_y"], row["ctr_z"]
        (z0, z1), (y0, y1), (x0, x1) = compute_bbox(cx, cy, cz, half_xy, half_z)
        try:
            subvol = vol[z0:z1, y0:y1, x0:x1]
            arr = np.squeeze(np.array(subvol))
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            # CloudVolume (Z, Y, X) -> SynClass (H, W, Z)
            arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
            shape = arr.shape
            syn_path = os.path.join(args.out_dir, f"{sid}_syn.npy")
            np.save(syn_path, arr)
            np.save(os.path.join(args.out_dir, f"{sid}_pre_syn_n_mask.npy"), np.zeros(shape, dtype=np.float32))
            np.save(os.path.join(args.out_dir, f"{sid}_post_syn_n_mask.npy"), np.zeros(shape, dtype=np.float32))
            done.append({"id_": sid, "pre_clf_type": row["pre_clf_type"]})
        except Exception as e:
            tqdm.write(f"Skip {sid}: {e}")

    out_csv = os.path.join(args.out_dir, "synapse_data.csv")
    pd.DataFrame(done).to_csv(out_csv, index=False)
    print(f"Downloaded {len(done)} synapses -> {args.out_dir}")
    print(f"Labels -> {out_csv}")
    print("Train with: python synapse_classifier_2dcnn_multichannel.py --csv_path", out_csv.replace("\\", "/"))


if __name__ == "__main__":
    main()
