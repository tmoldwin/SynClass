"""
Build manifest of synapses to download: proofread + known E/I (pre).
Output: Data/synapses_to_download.csv (id_, pre_clf_type, ctr_x, ctr_y, ctr_z).

Two modes:
- --use_cave: query CAVE synapse_query(pre_ids=...) in batches. No 47GB CSV needed.
- default: stream Data/synapses_pni_2.csv (requires download).
Requires: proofreading_status_public_release.csv, pre_root_id_to_ei.csv
"""

import argparse
import os

import pandas as pd


def load_proofread_roots(proof_csv: str) -> set[int]:
    df = pd.read_csv(proof_csv)
    if "pt_root_id" not in df.columns:
        df.columns = [
            "id", "valid", "pt_x", "pt_y", "pt_z", "pt_supervoxel_id",
            "pt_root_id", "valid_id", "status_dendrite", "status_axon"
        ]
    mask = df["status_axon"].isin(["clean", "extended"]) | df["status_dendrite"].isin(
        ["clean", "extended"]
    )
    return set(df.loc[mask, "pt_root_id"].astype("int64").tolist())


def load_ei_map(ei_csv: str) -> dict[int, str]:
    df = pd.read_csv(ei_csv)
    df = df[df["pre_clf_type"].isin(["E", "I"])]
    return dict(zip(df["pt_root_id"].astype("int64"), df["pre_clf_type"]))


def main():
    parser = argparse.ArgumentParser(
        description="Filter synapse table to proofread synapses with known E/I (pre)."
    )
    parser.add_argument(
        "--synapse_csv",
        default="Data/synapses_pni_2.csv",
        help="Full path to synapses_pni_2.csv (streamed in chunks).",
    )
    parser.add_argument(
        "--proof_csv",
        default="Data/proofreading_status_public_release.csv",
        help="Proofreading status CSV.",
    )
    parser.add_argument(
        "--ei_csv",
        default="Data/pre_root_id_to_ei.csv",
        help="Pre root_id -> E/I from fetch_pre_ei_map.py.",
    )
    parser.add_argument(
        "--out_csv",
        default="Data/synapses_to_download.csv",
        help="Output: id_, pre_clf_type, ctr_x, ctr_y, ctr_z.",
    )
    parser.add_argument(
        "--max_synapses",
        type=int,
        default=None,
        help="Max number of synapses to output (default: all).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500_000,
        help="Chunk size for streaming synapse CSV.",
    )
    parser.add_argument(
        "--use_cave",
        action="store_true",
        help="Query synapses from CAVE (synapse_query) instead of CSV. No 47GB download.",
    )
    parser.add_argument(
        "--pre_id_batch",
        type=int,
        default=100,
        help="When --use_cave: number of pre root IDs per CAVE query batch.",
    )
    args = parser.parse_args()

    for p in [args.proof_csv, args.ei_csv]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    if not args.use_cave and not os.path.isfile(args.synapse_csv):
        raise FileNotFoundError(
            f"Synapse CSV not found: {args.synapse_csv}. Download it or use --use_cave to query CAVE instead."
        )

    proof_roots = load_proofread_roots(args.proof_csv)
    ei_map = load_ei_map(args.ei_csv)
    print(f"Proofread roots: {len(proof_roots)}; pre E/I known: {len(ei_map)}")

    rows = []

    if args.use_cave:
        try:
            from caveclient import CAVEclient
        except ImportError:
            raise SystemExit("--use_cave requires: pip install caveclient (and token setup).")
        pre_with_ei = list(proof_roots & set(ei_map.keys()))
        print(f"Pre cells (proofread + E/I): {len(pre_with_ei)}")
        if not pre_with_ei:
            print("No proofread pre-synaptic cells with E/I. Check proof and E/I CSVs.")
        else:
            client = CAVEclient("minnie65_public")
            for i in range(0, len(pre_with_ei), args.pre_id_batch):
                batch = pre_with_ei[i : i + args.pre_id_batch]
                syn_df = client.materialize.synapse_query(
                    pre_ids=batch,
                    split_positions=True,
                )
                if syn_df is None or len(syn_df) == 0:
                    continue
                for _, r in syn_df.iterrows():
                    sid = int(r["id"])
                    pre_rid = int(r["pre_pt_root_id"])
                    pre_clf = ei_map.get(pre_rid)
                    if pre_clf not in ("E", "I"):
                        continue
                    pos = r.get("ctr_pt_position")
                    if pos is None and "ctr_pt_position_x" in r.index:
                        cx = r["ctr_pt_position_x"]
                        cy = r["ctr_pt_position_y"]
                        cz = r["ctr_pt_position_z"]
                    elif hasattr(pos, "__len__") and len(pos) >= 3:
                        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
                    else:
                        continue
                    rows.append({"id_": sid, "pre_clf_type": pre_clf, "ctr_x": cx, "ctr_y": cy, "ctr_z": cz})
                    if args.max_synapses is not None and len(rows) >= args.max_synapses:
                        break
                if args.max_synapses is not None and len(rows) >= args.max_synapses:
                    break
    else:
        first_chunk = pd.read_csv(args.synapse_csv, nrows=1)
        if "ctr_pt_position_x" in first_chunk.columns:
            pos_cols = ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
        elif "ctr_pt_position" in first_chunk.columns:
            pos_cols = ["ctr_pt_position"]
        else:
            raise ValueError("synapses_pni_2.csv must have ctr_pt_position or ctr_pt_position_x/y/z")
        usecols = ["id", "pre_pt_root_id", "post_pt_root_id"] + pos_cols
        total = 0
        for chunk in pd.read_csv(args.synapse_csv, usecols=usecols, chunksize=args.chunk_size):
            pre_ok = chunk["pre_pt_root_id"].isin(proof_roots)
            post_ok = chunk["post_pt_root_id"].isin(proof_roots)
            proofread = chunk[pre_ok | post_ok]
            pre_has_ei = proofread["pre_pt_root_id"].isin(ei_map)
            sub = proofread[pre_has_ei]
            for _, r in sub.iterrows():
                sid = int(r["id"])
                pre_clf = ei_map[int(r["pre_pt_root_id"])]
                if len(pos_cols) == 3:
                    cx, cy, cz = r["ctr_pt_position_x"], r["ctr_pt_position_y"], r["ctr_pt_position_z"]
                else:
                    pos = r["ctr_pt_position"]
                    if isinstance(pos, str):
                        pos = [float(x) for x in pos.strip("[]").split()]
                    cx, cy, cz = pos[0], pos[1], pos[2]
                rows.append({"id_": sid, "pre_clf_type": pre_clf, "ctr_x": cx, "ctr_y": cy, "ctr_z": cz})
                total += 1
                if args.max_synapses is not None and total >= args.max_synapses:
                    break
            if args.max_synapses is not None and total >= args.max_synapses:
                break

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} synapses -> {args.out_csv}")
    if len(out_df):
        print(out_df["pre_clf_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
