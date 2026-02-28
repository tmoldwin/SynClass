"""
Add pre_pt_root_id and post_pt_root_id to an existing manifest CSV by querying CAVE.
Overwrites the manifest in place (backs up original first).
"""
import argparse
import os
import shutil
import time

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Add root IDs to manifest CSV from CAVE synapse table.")
    parser.add_argument("--manifest", default="Data/synapses_to_download.csv")
    parser.add_argument("--batch_size", type=int, default=200, help="IDs per CAVE query")
    args = parser.parse_args()

    from caveclient import CAVEclient
    client = CAVEclient("minnie65_public")

    df = pd.read_csv(args.manifest)
    if "pre_pt_root_id" in df.columns and "post_pt_root_id" in df.columns:
        n_missing = df["pre_pt_root_id"].isna().sum()
        if n_missing == 0:
            print("Manifest already has root IDs for all rows.")
            return
        print(f"Manifest has root IDs but {n_missing} are missing, filling those.")
    else:
        df["pre_pt_root_id"] = pd.NA
        df["post_pt_root_id"] = pd.NA

    all_ids = df["id_"].tolist()
    id_to_roots = {}

    for i in tqdm(range(0, len(all_ids), args.batch_size), desc="Querying CAVE"):
        batch = all_ids[i:i + args.batch_size]
        # Only query IDs we don't have yet
        batch = [sid for sid in batch if sid not in id_to_roots]
        if not batch:
            continue
        for attempt in range(5):
            try:
                result = client.materialize.query_table(
                    "synapses_pni_2",
                    filter_in_dict={"id": batch},
                )
                for _, r in result.iterrows():
                    id_to_roots[int(r["id"])] = (int(r["pre_pt_root_id"]), int(r["post_pt_root_id"]))
                break
            except Exception as e:
                if attempt < 4:
                    wait = 2 ** attempt
                    tqdm.write(f"Retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    tqdm.write(f"Failed batch starting at {i}: {e}")

    matched = 0
    for idx, row in df.iterrows():
        sid = int(row["id_"])
        if sid in id_to_roots:
            pre_r, post_r = id_to_roots[sid]
            df.at[idx, "pre_pt_root_id"] = pre_r
            df.at[idx, "post_pt_root_id"] = post_r
            matched += 1

    # Backup and save
    backup = args.manifest + ".bak"
    if not os.path.isfile(backup):
        shutil.copy2(args.manifest, backup)
        print(f"Backed up to {backup}")

    df.to_csv(args.manifest, index=False)
    n_missing = df["pre_pt_root_id"].isna().sum()
    print(f"Updated {matched}/{len(df)} rows. Missing root IDs: {n_missing}")


if __name__ == "__main__":
    main()
