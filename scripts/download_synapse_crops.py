"""
Download EM cutouts for synapses listed in synapses_to_download.csv.
Saves volumes in SynClass format: {id}_syn.npy, {id}_pre_syn_n_mask.npy, {id}_post_syn_n_mask.npy
(masks are zeros; SynClass expects 3-channel input). Writes synapse_data.csv (id_, pre_clf_type).

Uses ImageryClient + CAVE by default so every synapse gets an image (full volume access).
Fallback: CloudVolume direct (bossdb slab only) if --no_imagery_client.

Requires: pip install pandas numpy tqdm
With ImageryClient (default): pip install caveclient imageryclient
Fallback: pip install cloud-volume
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


EM_CLOUDPATH = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"


def to_em_voxels(cx, cy, cz):
    """Convert 4,4,40 voxel coords to 8,8,40."""
    return int(cx) // 2, int(cy) // 2, int(cz)


def compute_bbox(cx, cy, cz, half_xy, half_z):
    """Return (z0, z1), (y0, y1), (x0, x1) in 8,8,40 voxel indices."""
    cx, cy, cz = int(cx), int(cy), int(cz)
    z0, z1 = max(cz - half_z, 0), cz + half_z
    y0, y1 = max(cy - half_xy, 0), cy + half_xy
    x0, x1 = max(cx - half_xy, 0), cx + half_xy
    return (z0, z1), (y0, y1), (x0, x1)


def download_via_imageryclient(df, args):
    """Use CAVE + ImageryClient so every synapse gets an image + real segmentation masks."""
    from caveclient import CAVEclient
    import imageryclient as ic

    client = CAVEclient("minnie65_public")
    img_client = ic.ImageryClient(client=client)
    bbox_xy = args.cube_xy
    bbox_z = args.cube_z
    has_root_ids = "pre_pt_root_id" in df.columns and "post_pt_root_id" in df.columns
    done = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        sid = int(row["id_"])
        ctr = [int(row["ctr_x"]), int(row["ctr_y"]), int(row["ctr_z"])]
        try:
            image, seg_dict = img_client.image_and_segmentation_cutout(
                ctr,
                split_segmentations=True,
                bbox_size=(bbox_xy, bbox_xy, bbox_z),
            )
            arr = np.squeeze(np.array(image))
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            arr = arr.astype(np.float32)
            shape = arr.shape

            # Build pre/post masks from segmentation using root IDs
            if has_root_ids:
                pre_root = int(row["pre_pt_root_id"])
                post_root = int(row["post_pt_root_id"])
                pre_mask_raw = seg_dict.get(pre_root)
                post_mask_raw = seg_dict.get(post_root)
                pre_mask = np.squeeze(np.array(pre_mask_raw)).astype(np.float32) if pre_mask_raw is not None else np.zeros(shape, dtype=np.float32)
                post_mask = np.squeeze(np.array(post_mask_raw)).astype(np.float32) if post_mask_raw is not None else np.zeros(shape, dtype=np.float32)
                if pre_mask.ndim == 2:
                    pre_mask = pre_mask[np.newaxis, :, :]
                if post_mask.ndim == 2:
                    post_mask = post_mask[np.newaxis, :, :]
            else:
                pre_mask = np.zeros(shape, dtype=np.float32)
                post_mask = np.zeros(shape, dtype=np.float32)

            np.save(os.path.join(args.out_dir, f"{sid}_syn.npy"), arr)
            np.save(os.path.join(args.out_dir, f"{sid}_pre_syn_n_mask.npy"), pre_mask)
            np.save(os.path.join(args.out_dir, f"{sid}_post_syn_n_mask.npy"), post_mask)
            done.append({"id_": sid, "pre_clf_type": row["pre_clf_type"]})
        except Exception as e:
            tqdm.write(f"Skip {sid}: {e}")
    return done


def download_via_cloudvolume(df, args):
    """Use CloudVolume direct (limited to bossdb slab bounds)."""
    from cloudvolume import CloudVolume

    half_xy = (args.cube_xy // 2) // 2
    half_z = args.cube_z // 2
    vol = CloudVolume(EM_CLOUDPATH, mip=0, cache=False, progress=True, fill_missing=True)
    b = vol.bounds
    z_min, z_max = int(b.minpt[0]), int(b.maxpt[0])
    y_min, y_max = int(b.minpt[1]), int(b.maxpt[1])
    x_min, x_max = int(b.minpt[2]), int(b.maxpt[2])
    print(f"Volume bounds: x=[{x_min},{x_max}] y=[{y_min},{y_max}] z=[{z_min},{z_max}]")

    def in_bounds(cx, cy, cz):
        cx_em, cy_em, cz_em = to_em_voxels(cx, cy, cz)
        (z0, z1), (y0, y1), (x0, x1) = compute_bbox(cx_em, cy_em, cz_em, half_xy, half_z)
        return (
            z_min <= z0 and z1 <= z_max
            and y_min <= y0 and y1 <= y_max
            and x_min <= x0 and x1 <= x_max
        )

    df["_in_bounds"] = df.apply(
        lambda r: in_bounds(r["ctr_x"], r["ctr_y"], r["ctr_z"]), axis=1
    )
    df = df[df["_in_bounds"]].drop(columns=["_in_bounds"])
    if len(df) == 0:
        print("No synapses in slab bounds. Use default (ImageryClient) for full volume.")
        return []

    done = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        sid = int(row["id_"])
        cx_em, cy_em, cz_em = to_em_voxels(row["ctr_x"], row["ctr_y"], row["ctr_z"])
        (z0, z1), (y0, y1), (x0, x1) = compute_bbox(cx_em, cy_em, cz_em, half_xy, half_z)
        try:
            subvol = vol[z0:z1, y0:y1, x0:x1]
            arr = np.squeeze(np.array(subvol))
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
            shape = arr.shape
            np.save(os.path.join(args.out_dir, f"{sid}_syn.npy"), arr)
            np.save(
                os.path.join(args.out_dir, f"{sid}_pre_syn_n_mask.npy"),
                np.zeros(shape, dtype=np.float32),
            )
            np.save(
                os.path.join(args.out_dir, f"{sid}_post_syn_n_mask.npy"),
                np.zeros(shape, dtype=np.float32),
            )
            done.append({"id_": sid, "pre_clf_type": row["pre_clf_type"]})
        except Exception as e:
            tqdm.write(f"Skip {sid}: {e}")
    return done


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
        help="Crop size in x,y (voxels at 4,4,40).",
    )
    parser.add_argument(
        "--cube_z",
        type=int,
        default=64,
        help="Crop size in z (voxels).",
    )
    parser.add_argument(
        "--max_synapses",
        type=int,
        default=None,
        help="Max number to download (default: all in manifest).",
    )
    parser.add_argument(
        "--no_imagery_client",
        action="store_true",
        help="Use CloudVolume direct instead of ImageryClient (only synapses in bossdb slab).",
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

    if args.no_imagery_client:
        print("Using CloudVolume direct (slab bounds only)...")
        done = download_via_cloudvolume(df, args)
    else:
        try:
            print("Using ImageryClient + CAVE (full volume, every synapse gets an image)...")
            done = download_via_imageryclient(df, args)
        except ImportError as e:
            raise SystemExit(
                "ImageryClient path requires: pip install caveclient imageryclient\n"
                "Or run with --no_imagery_client to use CloudVolume (slab only)."
            ) from e

    out_csv = os.path.join(args.out_dir, "synapse_data.csv")
    pd.DataFrame(done).to_csv(out_csv, index=False)
    print(f"Downloaded {len(done)}/{len(df)} synapses -> {args.out_dir}")
    print(f"Labels -> {out_csv}")
    if done:
        print("Train with: python synapse_classifier_2dcnn_multichannel.py --csv_path", out_csv.replace("\\", "/"))


if __name__ == "__main__":
    main()
