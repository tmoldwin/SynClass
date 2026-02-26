"""
Build a training CSV that keeps only proofread synapses with E/I labels.

Uses:
- proofreading_status_public_release.csv  → which cells are hand-proofread
- synapse_data.csv (existing)            → id_, pre_clf_type (E/I), pre_id, post_id

Keeps only synapses where at least one partner (pre or post) is proofread.
Output: proofread_synapse_data.csv with columns id_, pre_clf_type for training.

No need for the 47 GB synapses_pni_2.csv; your existing CSV already has root IDs.
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Build proofread-only E/I training CSV from existing synapse_data + proofreading status."
    )
    parser.add_argument(
        "--proof_csv",
        default="Data/proofreading_status_public_release.csv",
        help="Path to proofreading_status_public_release.csv",
    )
    parser.add_argument(
        "--synapse_csv",
        default="Data/synapse_data.csv",
        help="Path to synapse_data.csv (must have id_, pre_clf_type, pre_id, post_id)",
    )
    parser.add_argument(
        "--data_dir",
        default="Data/synpase_raw_em",
        help="Directory with *_syn.npy files; only output IDs that have volumes here.",
    )
    parser.add_argument(
        "--out_csv",
        default="Data/synpase_raw_em/proofread_synapse_data.csv",
        help="Output CSV path (id_, pre_clf_type only).",
    )
    parser.add_argument(
        "--no_filter_files",
        action="store_true",
        help="If set, do not filter to IDs that exist as *_syn.npy in data_dir.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.proof_csv):
        raise FileNotFoundError(
            f"Proofreading CSV not found: {args.proof_csv}\n"
            "Download it:\n"
            '  curl.exe -L "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/proofreading_status/proofreading_status_public_release.csv" -o data/proofreading_status_public_release.csv'
        )
    if not os.path.isfile(args.synapse_csv):
        raise FileNotFoundError(
            f"Synapse CSV not found: {args.synapse_csv}\n"
            "Use your existing synapse_data.csv with id_, pre_clf_type, pre_id, post_id."
        )

    # Proofread root IDs (cells with clean/extended axon or dendrite)
    proof = pd.read_csv(args.proof_csv)
    if "pt_root_id" not in proof.columns:
        proof.columns = [
            "id", "valid", "pt_x", "pt_y", "pt_z", "pt_supervoxel_id",
            "pt_root_id", "valid_id", "status_dendrite", "status_axon"
        ]
    mask = proof["status_axon"].isin(["clean", "extended"]) | proof["status_dendrite"].isin(
        ["clean", "extended"]
    )
    proof_roots = set(proof.loc[mask, "pt_root_id"].astype("int64").tolist())
    print(f"Proofread root IDs: {len(proof_roots)}")

    # Synapses with E/I and pre/post root IDs
    syn = pd.read_csv(args.synapse_csv)
    for col in ["id_", "pre_clf_type", "pre_id", "post_id"]:
        if col not in syn.columns:
            raise ValueError(f"synapse_data.csv must have column '{col}'")
    syn["pre_id"] = syn["pre_id"].astype("int64")
    syn["post_id"] = syn["post_id"].astype("int64")

    # Keep only E/I and only proofread partners
    syn = syn[syn["pre_clf_type"].isin(["E", "I"])]
    pre_ok = syn["pre_id"].isin(proof_roots)
    post_ok = syn["post_id"].isin(proof_roots)
    proofread = syn[pre_ok | post_ok].copy()
    print(f"Proofread synapses (with E/I): {len(proofread)} (of {len(syn)} total)")

    if not args.no_filter_files and os.path.isdir(args.data_dir):
        existing = set()
        for f in os.listdir(args.data_dir):
            if f.endswith("_syn.npy"):
                stem = f.replace("_syn.npy", "")
                if stem.isdigit():
                    existing.add(int(stem))
        proofread = proofread[proofread["id_"].isin(existing)]
        print(f"With existing volumes in {args.data_dir}: {len(proofread)}")
    else:
        print("Skipping file filter (--no_filter_files or data_dir missing).")

    out = proofread[["id_", "pre_clf_type"]].drop_duplicates(subset=["id_"])
    if out.duplicated("id_").any():
        out = out.drop_duplicates(subset=["id_"], keep="first")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} rows -> {args.out_csv}")
    print("Train on proofread-only data:")
    print(f"  python synapse_classifier_2dcnn_multichannel.py --csv_path {args.out_csv} --epochs 50 --batch_size 32")


if __name__ == "__main__":
    main()
