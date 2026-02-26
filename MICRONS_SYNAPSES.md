## MICrONS minnie65 synapses and hand-proofread cells

This repo already contains the key CSVs you need:

- **Synapse graph (all synapses, v117)**: `data/synapses_pni_2.csv`  
  - Source: `https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv`  
  - ~47.5 GB, one row per detected synapse with pre/post cell IDs and positions.
- **Proofreading status (which cells are hand-proofread)**: `data/proofreading_status_public_release.csv`  
  - Source: `https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/proofreading_status/proofreading_status_public_release.csv`

The electron microscopy (EM) imagery itself lives remotely:

- **EM volume (minnie65)**: `https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em`  
  (Multi-resolution precomputed volume; do *not* download the whole thing. Stream cutouts.)

---

## 1. Re-download the CSVs if needed

From the repo root (`SynClass`), on Windows PowerShell:

```powershell
# Synapse graph (47.5 GB, long download)
curl.exe -L "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/synapse_graph/synapses_pni_2.csv" `
  -o data/synapses_pni_2.csv

# Proofreading status table (small)
curl.exe -L "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/proofreading_status/proofreading_status_public_release.csv" `
  -o data/proofreading_status_public_release.csv
```

---

## 2. Python environment for pulling synapse “images”

Install the minimal dependencies (once):

```powershell
pip install cloud-volume pandas numpy tqdm
```

Key pieces:

- **EM volume**: accessed via `cloud-volume` using the precomputed path.  
- **Synapse table**: `synapses_pni_2.csv`, with:
  - `ctr_pt_position_x`, `ctr_pt_position_y`, `ctr_pt_position_z` (synapse center, in 4,4,40 nm voxels)
  - `pre_pt_root_id`, `post_pt_root_id` (cell IDs at v117)
- **Proofreading table**: `proofreading_status_public_release.csv`, with:
  - `status_axon`, `status_dendrite` in `{clean, extended, non}`
  - `pt_root_id` (cell root id at v117)

---

## 3. Script: sample EM cutouts around proofread synapses

Use this script to pull small 3D EM cubes (“synapse images”) around synapse centers, **restricted to synapses where at least one partner cell is hand-proofread**.

Save as `scripts/download_synapse_crops.py`:

```python
import argparse
import os

import numpy as np
import pandas as pd
from cloudvolume import CloudVolume
from tqdm import tqdm


EM_CLOUDPATH = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"


def load_proofread_root_ids(proof_csv: str) -> set[int]:
    df = pd.read_csv(proof_csv)
    # Cells with hand proofreading (axon and/or dendrite)
    mask = df["status_axon"].isin(["clean", "extended"]) | df["status_dendrite"].isin(
        ["clean", "extended"]
    )
    roots = df.loc[mask, "pt_root_id"].astype("int64")
    return set(roots.tolist())


def iter_proofread_synapses(syn_csv: str, proof_roots: set[int], max_synapses: int | None):
    usecols = [
        "id",
        "pre_pt_root_id",
        "post_pt_root_id",
        "ctr_pt_position_x",
        "ctr_pt_position_y",
        "ctr_pt_position_z",
    ]

    # Stream the huge CSV in chunks to avoid blowing up RAM
    total = 0
    for chunk in pd.read_csv(syn_csv, usecols=usecols, chunksize=200_000):
        # Restrict to synapses where either partner is proofread
        pre_ok = chunk["pre_pt_root_id"].isin(proof_roots)
        post_ok = chunk["post_pt_root_id"].isin(proof_roots)
        sub = chunk[pre_ok | post_ok]

        for _, row in sub.iterrows():
            yield row
            total += 1
            if max_synapses is not None and total >= max_synapses:
                return


def compute_bbox(center_xyz, half_size_xyz):
    cx, cy, cz = map(int, center_xyz)
    hx, hy, hz = half_size_xyz
    # CloudVolume slices are [z, y, x]
    z0, z1 = max(cz - hz, 0), cz + hz
    y0, y1 = max(cy - hy, 0), cy + hy
    x0, x1 = max(cx - hx, 0), cx + hx
    return (z0, z1), (y0, y1), (x0, x1)


def main():
    parser = argparse.ArgumentParser(description="Download EM cutouts around proofread synapses.")
    parser.add_argument("--synapse_csv", default="data/synapses_pni_2.csv")
    parser.add_argument("--proof_csv", default="data/proofreading_status_public_release.csv")
    parser.add_argument(
        "--out_dir",
        default="data/synapse_crops",
        help="Output directory for .npy volumes.",
    )
    parser.add_argument(
        "--cube_xy",
        type=int,
        default=256,
        help="Cube size in x and y (voxels) around each synapse center.",
    )
    parser.add_argument(
        "--cube_z",
        type=int,
        default=64,
        help="Cube size in z (voxels) around each synapse center.",
    )
    parser.add_argument(
        "--max_synapses",
        type=int,
        default=100,
        help="Maximum number of synapses to download (None for all).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading proofread root IDs...")
    proof_roots = load_proofread_root_ids(args.proof_csv)
    print(f"Found {len(proof_roots)} proofread cells.")

    print("Opening EM volume via CloudVolume...")
    vol = CloudVolume(EM_CLOUDPATH, mip=0, cache=False, progress=True, fill_missing=True)

    half_xy = args.cube_xy // 2
    half_z = args.cube_z // 2

    it = iter_proofread_synapses(args.synapse_csv, proof_roots, args.max_synapses)
    for row in tqdm(it, desc="Downloading synapse crops"):
        syn_id = int(row["id"])
        cx = row["ctr_pt_position_x"]
        cy = row["ctr_pt_position_y"]
        cz = row["ctr_pt_position_z"]
        (z0, z1), (y0, y1), (x0, x1) = compute_bbox(
            (cx, cy, cz), (half_xy, half_xy, half_z)
        )

        # Fetch EM subvolume [z, y, x, channel]
        subvol = vol[z0:z1, y0:y1, x0:x1]
        arr = np.array(subvol)

        out_path = os.path.join(args.out_dir, f"syn_{syn_id}_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}.npy")
        np.save(out_path, arr)


if __name__ == "__main__":
    main()
```

---

## 4. How to run the script

From the repo root:

```powershell
python scripts/download_synapse_crops.py `
  --synapse_csv data/synapses_pni_2.csv `
  --proof_csv data/proofreading_status_public_release.csv `
  --out_dir data/synapse_crops `
  --cube_xy 256 `
  --cube_z 64 `
  --max_synapses 100
```

Notes:

- **`--max_synapses`** keeps the run reasonable; increase once you’re happy.  
- Output volumes are saved as `.npy` arrays under `data/synapse_crops`, one file per synapse.  
- All synapses are chosen such that *at least one partner neuron* is marked `clean` or `extended` in the proofreading table.

This is the minimal, repeatable path from the official MICrONS static data to local, hand-proofread synapse EM cutouts.

