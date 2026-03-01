"""
Download synapse EM crops + pre/post segmentation masks from CAVE.

Only keeps z-slices where both pre and post masks are nonzero.
Synapses where neither root ID appears in the cutout are skipped entirely.

Reads manifest CSV (id_, pre_clf_type, ctr_x, ctr_y, ctr_z, pre_pt_root_id, post_pt_root_id).
Produces per-synapse:  {id}_syn.npy, {id}_pre_syn_n_mask.npy, {id}_post_syn_n_mask.npy
And a labels file:     synapse_data.csv
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm


def download_synapse(img_client, ctr, pre_root, post_root, bbox_size, z_subsample=5):
    """Download EM + masks, strip zero-mask slices, subsample every z_subsample-th from center.
    Returns None if no mask data found."""
    image, seg_dict = img_client.image_and_segmentation_cutout(
        ctr, split_segmentations=True, bbox_size=bbox_size,
    )
    em = np.squeeze(np.array(image)).astype(np.float32)
    if em.ndim == 2:
        em = em[:, :, np.newaxis]

    shape = em.shape
    pre_raw = seg_dict.get(pre_root)
    post_raw = seg_dict.get(post_root)

    pre_mask = np.squeeze(np.array(pre_raw)).astype(np.float32) if pre_raw is not None else np.zeros(shape, dtype=np.float32)
    post_mask = np.squeeze(np.array(post_raw)).astype(np.float32) if post_raw is not None else np.zeros(shape, dtype=np.float32)

    if pre_mask.ndim == 2:
        pre_mask = pre_mask[:, :, np.newaxis]
    if post_mask.ndim == 2:
        post_mask = post_mask[:, :, np.newaxis]

    # Pad masks to match EM shape if needed
    if pre_mask.shape != shape:
        pre_mask = np.zeros(shape, dtype=np.float32)
    if post_mask.shape != shape:
        post_mask = np.zeros(shape, dtype=np.float32)

    # Keep only z-slices where at least one mask is nonzero
    has_mask = np.any(pre_mask, axis=(0, 1)) & np.any(post_mask, axis=(0, 1))
    if not has_mask.any():
        return None

    em = em[:, :, has_mask]
    pre_mask = pre_mask[:, :, has_mask]
    post_mask = post_mask[:, :, has_mask]

    # Subsample: keep every 5th slice, centered on the middle
    n_valid = em.shape[2]
    if n_valid > 1:
        center = n_valid // 2
        indices = list(range(center, -1, -z_subsample)) [::-1] + list(range(center + z_subsample, n_valid, z_subsample))
        em = em[:, :, indices]
        pre_mask = pre_mask[:, :, indices]
        post_mask = post_mask[:, :, indices]

    return em, pre_mask, post_mask


def _download_one(args_tuple):
    """Worker: download one synapse. Returns ('ok'|'skip'|'no_mask'|'fail', sid, pre_clf_type or None)."""
    (row, out_dir, bbox_size, z_subsample, skip_existing, max_retries) = args_tuple
    sid = int(row["id_"])
    syn_path = os.path.join(out_dir, f"{sid}_syn.npy")

    if skip_existing and os.path.exists(syn_path):
        return ("skip", sid, row["pre_clf_type"])

    from caveclient import CAVEclient
    import imageryclient as ic
    client = CAVEclient("minnie65_public")
    img_client = ic.ImageryClient(client=client)

    ctr = [int(row["ctr_x"]), int(row["ctr_y"]), int(row["ctr_z"])]
    pre_root = int(row["pre_pt_root_id"])
    post_root = int(row["post_pt_root_id"])

    for attempt in range(max_retries):
        try:
            result = download_synapse(img_client, ctr, pre_root, post_root, bbox_size, z_subsample)
            break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ("fail", sid, None)

    if result is None:
        return ("no_mask", sid, None)

    em, pre_mask, post_mask = result
    np.save(syn_path, em)
    np.save(os.path.join(out_dir, f"{sid}_pre_syn_n_mask.npy"), pre_mask)
    np.save(os.path.join(out_dir, f"{sid}_post_syn_n_mask.npy"), post_mask)
    return ("ok", sid, row["pre_clf_type"])


def main():
    parser = argparse.ArgumentParser(description="Download synapse EM + masks (nonzero-mask slices only)")
    parser.add_argument("--manifest_csv", default="Data/synapses_to_download.csv")
    parser.add_argument("--out_dir", default="Data/proofread_synapses")
    parser.add_argument("--bbox_xy", type=int, default=256,
                        help="Crop size in x,y (voxels at synapse coordinate resolution)")
    parser.add_argument("--bbox_z", type=int, default=64, help="Crop size in z")
    parser.add_argument("--z_subsample", type=int, default=5,
                        help="Keep every Nth nonzero-mask slice, centered on middle")
    parser.add_argument("--max_per_class", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel download workers (default 6; more = faster but don't overload CAVE)")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip synapses that already have files on disk")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    for col in ("id_", "ctr_x", "ctr_y", "ctr_z", "pre_pt_root_id", "post_pt_root_id", "pre_clf_type"):
        assert col in df.columns, f"Manifest missing column: {col}"

    e_df = df[df["pre_clf_type"] == "E"].head(args.max_per_class)
    i_df = df[df["pre_clf_type"] == "I"].head(args.max_per_class)
    df = pd.concat([e_df, i_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Manifest: {len(e_df)} E + {len(i_df)} I = {len(df)} synapses")

    os.makedirs(args.out_dir, exist_ok=True)
    bbox_size = (args.bbox_xy, args.bbox_xy, args.bbox_z)

    work = []
    for _, row in df.iterrows():
        syn_path = os.path.join(args.out_dir, f"{int(row['id_'])}_syn.npy")
        if args.skip_existing and os.path.exists(syn_path):
            continue  # will not be in work; count these separately
        work.append((row, args.out_dir, bbox_size, args.z_subsample, args.skip_existing, args.max_retries))

    skipped_existing = len(df) - len(work)
    done = []
    skipped_no_mask = 0
    failed = 0

    print(f"Downloading {len(work)} synapses with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_download_one, item): item for item in work}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            status, sid, pre_clf = fut.result()
            if status == "ok":
                done.append({"id_": sid, "pre_clf_type": pre_clf})
            elif status == "skip":
                done.append({"id_": sid, "pre_clf_type": pre_clf})
            elif status == "no_mask":
                skipped_no_mask += 1
            else:
                failed += 1
                tqdm.write(f"FAIL {sid}")

    # Include skipped (already on disk) in done so CSV is complete
    if args.skip_existing and skipped_existing > 0:
        existing_ids = {int(f.replace("_syn.npy", "")) for f in os.listdir(args.out_dir) if f.endswith("_syn.npy")}
        for _, row in df.iterrows():
            if int(row["id_"]) in existing_ids and not any(d["id_"] == int(row["id_"]) for d in done):
                done.append({"id_": int(row["id_"]), "pre_clf_type": row["pre_clf_type"]})

    out_csv = os.path.join(args.out_dir, "synapse_data.csv")
    pd.DataFrame(done).to_csv(out_csv, index=False)

    e_done = sum(1 for d in done if d["pre_clf_type"] == "E")
    i_done = sum(1 for d in done if d["pre_clf_type"] == "I")
    print(f"\nDone: {len(done)} saved (E={e_done}, I={i_done})")
    print(f"  {skipped_existing} skipped (already on disk)")
    print(f"  {skipped_no_mask} skipped (no mask data in cutout)")
    print(f"  {failed} failed (API errors)")


if __name__ == "__main__":
    main()
