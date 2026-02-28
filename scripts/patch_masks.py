"""
Patch mask files for existing synapse crops: fetch real pre/post segmentation
masks from CAVE and overwrite the zero-mask .npy files in place.
Does NOT re-fetch EM images. Matches bbox to each synapse's existing shape.

Requires manifest CSV with: id_, ctr_x, ctr_y, ctr_z, pre_pt_root_id, post_pt_root_id
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Patch zero masks with real segmentation from CAVE.")
    parser.add_argument("--data_dir", default="Data/proofread_synapses", help="Dir with *_syn.npy files")
    parser.add_argument("--manifest_csv", default="Data/synapses_to_download.csv",
                        help="CSV with id_, ctr_x, ctr_y, ctr_z, pre_pt_root_id, post_pt_root_id")
    parser.add_argument("--max_synapses", type=int, default=None, help="Limit (for testing)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if mask file already has nonzero data")
    args = parser.parse_args()

    from caveclient import CAVEclient
    import imageryclient as ic

    client = CAVEclient("minnie65_public")
    img_client = ic.ImageryClient(client=client)

    df = pd.read_csv(args.manifest_csv)
    for col in ("id_", "ctr_x", "ctr_y", "ctr_z", "pre_pt_root_id", "post_pt_root_id"):
        if col not in df.columns:
            raise ValueError(f"Manifest missing column: {col}. Re-run filter_synapses_ei.py to get root IDs.")

    # Only patch synapses that have an existing *_syn.npy on disk
    existing_ids = set()
    for f in os.listdir(args.data_dir):
        if f.endswith("_syn.npy"):
            try:
                existing_ids.add(int(f.replace("_syn.npy", "")))
            except ValueError:
                pass
    df = df[df["id_"].isin(existing_ids)]
    if args.max_synapses:
        df = df.head(args.max_synapses)
    print(f"Patching masks for {len(df)} synapses (out of {len(existing_ids)} on disk)")

    patched = 0
    skipped = 0
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Patching masks"):
        sid = int(row["id_"])
        syn_path = os.path.join(args.data_dir, f"{sid}_syn.npy")
        pre_path = os.path.join(args.data_dir, f"{sid}_pre_syn_n_mask.npy")
        post_path = os.path.join(args.data_dir, f"{sid}_post_syn_n_mask.npy")

        try:
            vol = np.load(syn_path)
        except (ValueError, OSError):
            failed += 1
            continue
        if vol.size == 0:
            failed += 1
            continue

        if args.skip_existing:
            try:
                pm = np.load(pre_path)
                if pm.any():
                    skipped += 1
                    continue
            except (FileNotFoundError, ValueError, OSError):
                pass

        # Use the on-disk volume shape to set bbox_size so masks match exactly
        # SynClass convention: (H, W, Z) -> bbox_size = (H, W, Z)
        shape = vol.shape
        bbox_size = (shape[0], shape[1], shape[2])
        ctr = [int(row["ctr_x"]), int(row["ctr_y"]), int(row["ctr_z"])]
        pre_root = int(row["pre_pt_root_id"])
        post_root = int(row["post_pt_root_id"])

        for attempt in range(3):
            try:
                _, seg_dict = img_client.image_and_segmentation_cutout(
                    ctr, split_segmentations=True, bbox_size=bbox_size,
                )
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    tqdm.write(f"Failed {sid}: {e}")
                    failed += 1
                    seg_dict = None

        if seg_dict is None:
            continue

        pre_raw = seg_dict.get(pre_root)
        post_raw = seg_dict.get(post_root)
        pre_mask = np.squeeze(np.array(pre_raw)).astype(np.float32) if pre_raw is not None else np.zeros(shape, dtype=np.float32)
        post_mask = np.squeeze(np.array(post_raw)).astype(np.float32) if post_raw is not None else np.zeros(shape, dtype=np.float32)

        # Ensure shape matches the EM volume exactly
        if pre_mask.shape != shape:
            pre_mask = np.zeros(shape, dtype=np.float32)
        if post_mask.shape != shape:
            post_mask = np.zeros(shape, dtype=np.float32)

        np.save(pre_path, pre_mask)
        np.save(post_path, post_mask)
        patched += 1

    print(f"Done: {patched} patched, {skipped} skipped (already had masks), {failed} failed")


if __name__ == "__main__":
    main()
